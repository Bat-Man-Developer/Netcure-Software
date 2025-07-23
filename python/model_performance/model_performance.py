import os
import random
import time
import json
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, make_scorer, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class NovelHybridModel:
    def __init__(self, data_paths, stability_threshold=0.5, n_iterations=10, subsample_size=0.7, n_subspaces=5, subspace_size=0.7):
        # Initialize class attributes
        self.data = self.load_data(data_paths)
        self.data = self.preprocess_data(self.data)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.split_data(self.data)
        
        self.stability_threshold = stability_threshold
        self.n_iterations = n_iterations
        self.subsample_size = subsample_size
        self.n_subspaces = n_subspaces
        self.subspace_size = subspace_size
        
        # Define dictionary of models with optimized parameters
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1),
            'KNN': KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
        }
        
        # Perform feature selection and model training
        self.feature_importance = self.stability_selection()
        self.trained_models = self.train_models()
        self.model_weights = self.calculate_model_weights()

    def load_data(self, data_paths):
        # Load data from multiple CSV files
        data_frames = []
        for path in data_paths:
            df = pd.read_csv(path, low_memory=False)
            data_frames.append(df)

        return pd.concat(data_frames, ignore_index=True)

    def preprocess_data(self, data):
        df = data.copy()
        
        # Drop unnecessary columns
        df = df.drop(['Unnamed: 0', 'Flow ID'], axis=1, errors='ignore')
        
        # Identify categorical columns
        categorical_columns = [' Protocol', ' Source IP', ' Destination IP', ' Label', ' SimillarHTTP', ' Inbound']
        
        # Convert timestamp to datetime
        df[' Timestamp'] = pd.to_datetime(df[' Timestamp'], errors='coerce')
        df[' Timestamp'] = df[' Timestamp'].fillna(df[' Timestamp'].median())
        
        # Identify numeric columns
        numeric_columns = [col for col in df.columns if col not in categorical_columns and col != ' Timestamp']
        
        # Handle numeric columns
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Impute numeric columns
        numeric_imputer = SimpleImputer(strategy='median')
        df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])
        
        # Scale numeric columns
        scaler = RobustScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Handle categorical columns
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        for col in categorical_columns:
            if col in df.columns:
                if col == ' Protocol':
                    df[col] = categorical_imputer.fit_transform(df[[col]]).ravel()
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif col in [' SimillarHTTP', ' Inbound']:
                    df[col] = categorical_imputer.fit_transform(df[[col]]).ravel()
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                else:
                    df[col] = categorical_imputer.fit_transform(df[[col]]).ravel()
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
        
        # Final cleanup
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())

        self.plot_data_analysis(df, data)

        return df
    
    def plot_data_analysis(self, df, data):
        # Create the model_plots directory if it doesn't exist
        os.makedirs('c:/Xampp/htdocs/net-cure-website/model_plots', exist_ok=True)

        # Exploratory Data Analysis Plots
        # 1. Temporal Analysis
        self.plot_temporal_analysis(df)

        # 2. Protocol Distribution
        self.plot_protocol_distribution(df)

        # 3. Correlation Heatmap for Numeric Features
        self.plot_feature_correlation(df)

        # 4. Flow Duration Analysis
        self.plot_flow_duration(df)

        # Statistical Modeling Plots
        # 1. Feature Distribution Analysis
        self.plot_feature_distributions(df, data)

        # 2. Attack Pattern Clustering
        self.plot_attack_clustering_PCA(df)

    def split_data(self, data):
        feature_columns = [col for col in data.columns if col != ' Label' and col != ' Timestamp']
        X = data[feature_columns]
        y = data[' Label'] if ' Label' in data.columns else None
            
        if y is None:
            raise ValueError("No target variable found in the dataset")
        
        # First split: temporary training and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
        
        # Second split: training and validation sets from temporary training set
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.30, random_state=42, stratify=y_temp)
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def tune_hfs_hyperparameters(self, cv=5):
        param_grids = {
            'IsolationForest': {
                'n_estimators': [100, 200, 300],
                'contamination': ['auto', 0.01, 0.02],
                'max_samples': ['auto', 256, 512, 1024],
                'max_features': [0.5, 0.7, 1.0],
                'bootstrap': [True, False],
                'n_jobs': [-1]
            },
            'DBSCAN': {
                'eps': np.logspace(-3, 0.5, 10),
                'min_samples': [15, 20, 25],
                'metric': ['euclidean', 'manhattan'],
                'algorithm': ['ball_tree', 'kd_tree'],
                'leaf_size': [40, 45, 50]
            },
            'RF_Method2': {
                'n_estimators': [100, 200, 300],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
                'class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 100}],
                'criterion': ['gini', 'entropy'],
                'n_jobs': [-1]
            },
            'GB_Method3': {
                'n_estimators': [100],
                'max_depth': [5, 7],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7],
                'min_samples_split': [10, 15],
                'min_samples_leaf': [1],
                'validation_fraction': [0.2],
                'n_iter_no_change': [10],
                'tol': [1e-4]
            }
        }
        
        best_hfs_params = {}
        
        scoring = {
            'recall': make_scorer(recall_score, average='weighted', zero_division=0),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'f1': make_scorer(f1_score, average='weighted', zero_division=0)
        }
        
        # Tune Isolation Forest
        print("\nTuning Isolation Forest")
        try:
            best_score = float('-inf')
            best_iso_params = None
            
            param_combinations = list(ParameterGrid(param_grids['IsolationForest']))
            random_combinations = random.sample(param_combinations, min(30, len(param_combinations)))
            
            for params in random_combinations:
                iso = IsolationForest(**params, random_state=42)
                iso.fit(self.X_train)
                scores = iso.decision_function(self.X_train)
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_iso_params = params
            
            if best_iso_params:
                best_hfs_params['IsolationForest'] = best_iso_params
                print("Best parameters for Isolation Forest:")
                print(best_iso_params)
                print(f"Best score: {best_score:.4f}")
                
                self.iso_forest = IsolationForest(**best_iso_params, random_state=42)
                self.iso_forest.fit(self.X_train)
                
        except Exception as e:
            print(f"Error tuning Isolation Forest: {str(e)}")
        
        # Tune DBSCAN
        print("\nTuning DBSCAN")
        try:
            best_silhouette = float('-inf')
            best_dbscan_params = None
            
            param_combinations = list(ParameterGrid(param_grids['DBSCAN']))
            random_combinations = random.sample(param_combinations, min(30, len(param_combinations)))
            
            for params in random_combinations:
                dbscan = DBSCAN(**params)
                labels = dbscan.fit_predict(self.X_train)
                
                if len(np.unique(labels)) > 1:
                    silhouette_avg = silhouette_score(self.X_train, labels)
                    
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_dbscan_params = params
            
            if best_dbscan_params:
                best_hfs_params['DBSCAN'] = best_dbscan_params
                print("Best parameters for DBSCAN:")
                print(best_dbscan_params)
                print(f"Best silhouette score: {best_silhouette:.4f}")
                
                self.dbscan = DBSCAN(**best_dbscan_params)
            
        except Exception as e:
            print(f"Error tuning DBSCAN: {str(e)}")
        
        # Tune RF and GB using RandomizedSearchCV
        for name in ['RF_Method2', 'GB_Method3']:
            print(f"\nTuning {name}")
            try:
                model = (RandomForestClassifier(random_state=42) if name == 'RF_Method2' 
                        else GradientBoostingClassifier(random_state=42))
                
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grids[name],
                    n_iter=8,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                    scoring=scoring,
                    refit='f1',
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
                    search.fit(self.X_train, self.y_train)
                
                best_hfs_params[name] = search.best_params_
                
                print(f"Best parameters for {name}:")
                print(json.dumps(best_hfs_params[name], indent=4))
                print("\nBest cross-validation scores:")
                for metric in scoring.keys():
                    score = search.cv_results_[f'mean_test_{metric}'][search.best_index_]
                    print(f"{metric}: {score:.4f}")
                
                if name == 'RF_Method2':
                    self.rf = RandomForestClassifier(**best_hfs_params[name], random_state=42)
                    self.rf.fit(self.X_train, self.y_train)
                else:
                    self.gb = GradientBoostingClassifier(**best_hfs_params[name], random_state=42)
                    self.gb.fit(self.X_train, self.y_train)
                    
            except Exception as e:
                print(f"Error tuning {name}: {str(e)}")

        print("\nFinal best parameters for all models:")
        print(json.dumps(best_hfs_params, indent=4))

        self.best_hfs_params = best_hfs_params
        return best_hfs_params
    
    def stability_selection(self):
        # Perform Hybrid Feature Selection Hyperparameter Tuning First
        self.tune_hfs_hyperparameters()

        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Data must be split before running stability selection")
        
        # Include validation data in the scaling process
        X_combined = np.vstack([
            self.X_train.values if hasattr(self.X_train, 'values') else self.X_train,
            self.X_val.values if hasattr(self.X_val, 'values') else self.X_val
        ])
        y_combined = np.concatenate([
            self.y_train.values if hasattr(self.y_train, 'values') else self.y_train,
            self.y_val.values if hasattr(self.y_val, 'values') else self.y_val
        ])
        
        # Initialize feature importance array
        feature_importance = np.zeros(X_combined.shape[1])
        
        # Robust scaling with outlier capping
        def robust_scale_with_caps(X, percentile_range=(1, 99)):
            X_capped = np.copy(X)
            for j in range(X.shape[1]):
                lower, upper = np.percentile(X[:, j], percentile_range)
                X_capped[:, j] = np.clip(X_capped[:, j], lower, upper)
            scaler = RobustScaler()
            return scaler.fit_transform(X_capped)
        
        X_scaled = robust_scale_with_caps(X_combined)
        
        feature_importance_iterations = []  # Store importance scores from each iteration
        
        for iteration in range(self.n_iterations):
            n_samples = X_scaled.shape[0]
            subsample_size_int = int(n_samples * self.subsample_size)
            
            # Stratified subsampling
            unique_classes = np.unique(y_combined)
            subsample_indices = []
            for cls in unique_classes:
                cls_indices = np.where(y_combined == cls)[0]
                cls_sample_size = int(subsample_size_int * len(cls_indices) / n_samples)
                subsample_indices.extend(
                    np.random.choice(cls_indices, size=cls_sample_size, replace=False)
                )
            
            X_subsample = X_scaled[subsample_indices]
            y_subsample = y_combined[subsample_indices]
            
            try:
                # Create class weights dictionary for all present classes
                unique_classes = np.unique(y_subsample)
                class_weights = dict(zip(unique_classes, 
                                    [1/len(np.where(y_subsample == c)[0]) 
                                        for c in unique_classes]))
                
                # 1. Isolation Forest for anomaly detection
                iso_forest = IsolationForest(**self.best_hfs_params['IsolationForest'], 
                                        random_state=42+iteration)  # Different seed per iteration
                iso_scores = iso_forest.fit_predict(X_subsample)
                normal_samples = iso_scores == 1
                
                # 2. DBSCAN for density-based clustering
                dbscan = DBSCAN(**self.best_hfs_params['DBSCAN'])
                cluster_labels = dbscan.fit_predict(X_subsample)
                valid_clusters = cluster_labels != -1
                
                # Combine filters with more lenient criteria for DDoS detection
                valid_samples = normal_samples | valid_clusters  # Changed & to | for higher sensitivity
                
                if np.sum(valid_samples) < 10:
                    valid_samples = np.ones_like(normal_samples, dtype=bool)
                
                # 3. Feature importance calculation using multiple methods
                importance_scores = np.zeros((3, X_scaled.shape[1]))
                
                # Method 1: Mutual Information with increased sensitivity
                for j in range(X_scaled.shape[1]):
                    importance_scores[0, j] = mutual_info_classif(
                        X_subsample[valid_samples][:, j:j+1], 
                        y_subsample[valid_samples], 
                        random_state=42+iteration,
                        n_neighbors=3  # Reduced for higher sensitivity
                    )[0]
                
                # Method 2: Random Forest with class weights
                rf_params = self.best_hfs_params['RF_Method2'].copy()
                rf_params['class_weight'] = class_weights
                rf = RandomForestClassifier(**rf_params, random_state=42+iteration)
                rf.fit(X_subsample[valid_samples], y_subsample[valid_samples])
                importance_scores[1, :] = rf.feature_importances_
                
                # Method 3: Gradient Boosting
                gb = GradientBoostingClassifier(
                    **self.best_hfs_params['GB_Method3'],
                    random_state=42+iteration
                )
                gb.fit(X_subsample[valid_samples], y_subsample[valid_samples])
                importance_scores[2, :] = gb.feature_importances_
                
                # Ensemble voting with adjusted weights for DDoS detection
                # Increased weight for Mutual Information as it's particularly good for detecting anomalies
                weights = np.array([0.4, 0.3, 0.3])  # Adjusted weights for MI, RF, and GB
                weighted_importance = np.average(importance_scores, axis=0, weights=weights)
                
                # Store importance scores for this iteration
                feature_importance_iterations.append(weighted_importance)
                
                # Update feature importance
                feature_importance += weighted_importance
                
            except Exception as e:
                print(f"Warning: Error in iteration {iteration}: {str(e)}")
                continue
        
        # Normalize feature importance scores
        feature_importance /= self.n_iterations
        
        # Calculate stability score based on consistency across iterations
        feature_importance_iterations = np.array(feature_importance_iterations)
        stability_scores = np.std(feature_importance_iterations, axis=0)
        normalized_stability = 1 - (stability_scores / np.max(stability_scores))
        
        # Combine importance and stability
        final_importance = feature_importance * normalized_stability
        
        # Enhanced Adaptive thresholding using Otsu's method
        sorted_importance = np.sort(final_importance)
        variances = np.zeros(len(sorted_importance)-1)
        
        for i in range(1, len(sorted_importance)):
            left = sorted_importance[:i]
            right = sorted_importance[i:]
            if len(left) > 0 and len(right) > 0:
                # Modified variance calculation with emphasis on between-class variance
                w1 = len(left) / len(sorted_importance)
                w2 = len(right) / len(sorted_importance)
                m1 = np.mean(left)
                m2 = np.mean(right)
                between_var = w1 * w2 * (m1 - m2) ** 2
                within_var = w1 * np.var(left) + w2 * np.var(right)
                variances[i-1] = within_var - (2 * between_var)  # Emphasis on separation
        
        optimal_idx = np.argmin(variances)
        self.stability_threshold = sorted_importance[optimal_idx]
        
        # Select features with importance above threshold
        self.stable_features = np.where(final_importance >= self.stability_threshold)[0]
        
        # Ensure minimum number of features with focus on DDoS detection
        min_features = max(10, int(0.15 * X_scaled.shape[1]))  # Increased minimum features
        if len(self.stable_features) < min_features:
            # Select additional features based on both importance and stability
            remaining_features = np.setdiff1d(np.arange(X_scaled.shape[1]), self.stable_features)
            additional_features = remaining_features[np.argsort(final_importance[remaining_features])[-min_features:]]
            self.stable_features = np.union1d(self.stable_features, additional_features)
        
        print(f"\nNo. of stable features: {len(self.stable_features)} out of {X_scaled.shape[1]}")
        print(f"Stable feature indices: {self.stable_features}")
        
        # Calculate and print feature importance statistics
        importance_stats = {
            'mean': np.mean(final_importance[self.stable_features]),
            'std': np.std(final_importance[self.stable_features]),
            'min': np.min(final_importance[self.stable_features]),
            'max': np.max(final_importance[self.stable_features])
        }
        print("\nFeature Importance Statistics for Selected Features:")
        for stat, value in importance_stats.items():
            print(f"{stat.capitalize()}: {value:.4f}")
        
        return final_importance
    
    def tune_hmlc_hyperparameters(self, param_grids=None, cv=5):
        if not hasattr(self, 'models'):
            raise ValueError("Models must be initialized before tuning HMLC hyperparameters")
            
        if not hasattr(self, 'stable_features'):
            raise ValueError("Stability selection must be performed before tuning HMLC hyperparameters")
        
        # Use only stable features for hmlc hyperparameter tuning
        X_train_stable = self.X_train.iloc[:, self.stable_features]
        
        if param_grids is None:
            param_grids = {
                'Decision Tree': {
                    'classifier__max_depth': [80, 85, 90, 95, 100],
                    'classifier__min_samples_split': [5, 10, 15],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 100}],
                    'classifier__min_impurity_decrease': [0.0, 0.01],
                    'classifier__random_state': [42]
                },
                'Random Forest': {
                    'classifier__n_estimators': [1000, 1500, 2000],
                    'classifier__max_depth': [25, 30, 35],
                    'classifier__min_samples_split': [2, 5, 7],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['sqrt', 'log2'],
                    'classifier__class_weight': ['balanced', {0: 1, 1: 10}, {0: 1, 1: 100}],
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__n_jobs': [-1],
                    'classifier__random_state': [42]
                },
                'KNN': {
                    'classifier__n_neighbors': [5, 10, 20],
                    'classifier__weights': ['distance'],
                    'classifier__algorithm': ['ball_tree', 'kd_tree'],
                    'classifier__leaf_size': [80, 85, 90],
                    'classifier__metric': ['euclidean', 'manhattan'],
                    'classifier__n_jobs': [-1]
                }
            }

        best_hmlc_params = {}
        tuned_models = {}
        
        scoring = {
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
            'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
        }

        if len(np.unique(self.y_train)) == 2:
            scoring['auc'] = 'roc_auc'
        
        for name, base_model in self.models.items():
            if name in param_grids:
                print(f"\nTuning hyperparameters for {name} using {len(self.stable_features)} stable features")
                try:
                    pipeline_steps = []
                    if hasattr(self, 'scaler') and self.scaler is not None:
                        pipeline_steps.append(('scaler', clone(self.scaler)))
                    pipeline_steps.append(('classifier', clone(base_model)))
                    pipeline = Pipeline(pipeline_steps)
                    
                    if len(pipeline_steps) > 1:
                        tuning_params = {f'classifier__{k}': v 
                                    for k, v in param_grids[name].items()}
                    else:
                        tuning_params = param_grids[name]
                    
                    # Use RandomizedSearchCV
                    search = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=tuning_params,
                        n_iter=36,
                        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
                        scoring=scoring,
                        refit='f1_weighted',
                        n_jobs=-1,
                        verbose=1,
                        random_state=42,
                        error_score='raise'
                    )
                    
                    search.fit(X_train_stable, self.y_train)
                    
                    best_hmlc_params[name] = {k.replace('classifier__', ''): v 
                                    for k, v in search.best_params_.items()}
                    tuned_models[name] = search.best_estimator_.named_steps['classifier']
                    
                    print(f"Best parameters for {name}:")
                    print(best_hmlc_params[name])
                    print(f"Best cross-validation scores:")
                    for metric in scoring.keys():
                        score_key = f'mean_test_{metric}'
                        mean_score = search.cv_results_[score_key][search.best_index_]
                        print(f"{metric}: {mean_score:.4f}")
                    
                except Exception as e:
                    print(f"Error tuning {name}: {str(e)}")
                    print(f"Using default parameters for {name}")
                    best_hmlc_params[name] = {}
                    tuned_models[name] = base_model
            else:
                print(f"\nNo parameter grid specified for {name}, using default parameters")
                best_hmlc_params[name] = {}
                tuned_models[name] = base_model
        
        self.models = tuned_models
        self.best_hmlc_params = best_hmlc_params
        
        return best_hmlc_params

    def random_subspace(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if not hasattr(self, 'stable_features'):
            raise ValueError("Must run stability_selection before random_subspace")
        
        n_stable_features = len(self.stable_features)
        subspace_size = max(1, int(n_stable_features * self.subspace_size))
        
        # Add error checking for feature dimensionality
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(f"Input X should have {self.X_train.shape[1]} features, but got {X.shape[1]}")
    
        selected_indices = np.random.choice(
            n_stable_features, 
            min(subspace_size, n_stable_features), 
            replace=False
        )
        selected_features = self.stable_features[selected_indices]
        
        return X[:, selected_features], selected_features
    
    def train_models(self, n_splits=10):
        # Record start time for overall training
        total_start_time = time.time()
        
        # Perform Hybrid Machine Learning Classifier Hyperparameter Tuning First
        self.tune_hmlc_hyperparameters()
        
        if not hasattr(self, 'stable_features'):
            raise ValueError("Must run stability_selection before training models")

        trained_models = {}
        best_models = {}
        model_training_times = {}
        
        # Initialize k-fold cross validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            # Record start time for this model
            model_start_time = time.time()
            
            best_accuracy = 0
            best_subspace_models = None
            consecutive_decreases = 0
            max_decreases = 3  # Stop after 3 consecutive decreases in accuracy
            
            # Keep track of validation accuracies
            val_accuracies = []
            
            for iteration in range(self.n_subspaces):
                subspace_models = []
                fold_accuracies = []
                
                try:
                    X_subspace, selected_features = self.random_subspace(self.X_train)
                    if hasattr(X_subspace, 'values'):
                        X_subspace = X_subspace.values
                    X_val_subspace = self.X_val.iloc[:, selected_features].values
                    
                    # Perform k-fold cross validation
                    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_subspace, self.y_train)):
                        X_fold_train = X_subspace[train_idx]
                        y_fold_train = self.y_train.iloc[train_idx]
                        X_fold_val = X_subspace[val_idx]
                        y_fold_val = self.y_train.iloc[val_idx]
                        
                        if name in self.best_hmlc_params:
                            subspace_model = clone(model).set_params(**self.best_hmlc_params[name])
                        else:
                            subspace_model = clone(model)
                        subspace_model.fit(X_fold_train, y_fold_train)
                        
                        # Calculate accuracy for this fold
                        y_pred = subspace_model.predict(X_fold_val)
                        fold_accuracy = accuracy_score(y_fold_val, y_pred)
                        fold_accuracies.append(fold_accuracy)
                        
                        subspace_models.append((subspace_model, selected_features))
                    
                    # Calculate average accuracy across folds
                    avg_fold_accuracy = np.mean(fold_accuracies)

                    print(f"\n{name}")
                    print(f"Average Fold Accuracy: {avg_fold_accuracy}")
                    
                    # Evaluate on validation set
                    final_model = clone(model)
                    final_model.fit(X_subspace, self.y_train)
                    y_val_pred = final_model.predict(X_val_subspace)
                    val_accuracy = accuracy_score(self.y_val, y_val_pred)
                    val_accuracies.append(val_accuracy)
                    
                    # Check if accuracy has improved
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_subspace_models = subspace_models.copy()
                        consecutive_decreases = 0
                    else:
                        consecutive_decreases += 1
                    
                    # Stop if accuracy has decreased consecutively
                    if consecutive_decreases >= max_decreases:
                        print(f"Stopping {name} training: Accuracy hasn't improved for {max_decreases} iterations")
                        break
                    
                except Exception as e:
                    print(f"Error training {name} on iteration {iteration}: {str(e)}")
                    continue
            
            # Record training time for this model
            model_training_time = time.time() - model_start_time
            model_training_times[name] = model_training_time
            print(f"{name} Training Time: {model_training_time:.2f} seconds")
            
            if best_subspace_models:
                trained_models[name] = best_subspace_models
                best_models[name] = {
                    'accuracy': best_accuracy,
                    'validation_accuracies': val_accuracies
                }
            else:
                print(f"Warning: No successful training for {name}")
        
        # Record total training time
        total_training_time = time.time() - total_start_time
        print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
        
        if not trained_models:
            raise ValueError("No models were successfully trained")
        
        # Print best validation accuracies for each model
        print("\nBest validation accuracies:")
        for name, info in best_models.items():
            print(f"{name}: {info['accuracy']:.4f}")
        
        self.model_training_times = model_training_times

        return trained_models

    def calculate_model_weights(self):
        weights = {}
        model_evidence = {}
        prior_probs = {name: 1.0 / len(self.trained_models) for name in self.trained_models}
        epsilon = 1e-10
        
        for name, subspace_models in self.trained_models.items():
            log_evidences = []
            for model, selected_features in subspace_models:
                X_test_subspace = self.X_test.values[:, selected_features]
                
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_subspace)
                        y_pred_proba = np.clip(y_pred_proba, epsilon, 1.0 - epsilon)
                        log_likelihood = np.sum([np.log(proba[min(true_class, len(proba)-1)]) 
                                        for proba, true_class in zip(y_pred_proba, self.y_test)])
                    else:
                        y_pred = model.predict(X_test_subspace)
                        log_likelihood = np.sum(np.log((y_pred == self.y_test).astype(float) + epsilon))
                    
                    n_params = len(selected_features)
                    n_samples = len(self.y_test)
                    bic_penalty = -0.5 * n_params * np.log(n_samples)
                    
                    log_likelihood = log_likelihood / n_samples
                    log_evidences.append(log_likelihood + bic_penalty)
                    
                except Exception as e:
                    print(f"Error processing model {name}: {str(e)}")
                    log_evidences.append(-np.inf)
            
            valid_evidences = [e for e in log_evidences if not np.isinf(e) and not np.isnan(e)]
            model_evidence[name] = np.mean(valid_evidences) if valid_evidences else -np.inf
        
        temperature = 1.0
        scaled_evidences = {name: evidence/temperature for name, evidence in model_evidence.items()}
        max_evidence = max(scaled_evidences.values())
        
        for name in scaled_evidences:
            weights[name] = np.exp(scaled_evidences[name] - max_evidence) * prior_probs[name]
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w/total_weight for name, w in weights.items()}
        else:
            weights = prior_probs
        
        print("\nModel Evidences:")
        for name, evidence in model_evidence.items():
            print(f"{name}: {evidence:.4f}")
        
        print("\nFinal Model Weights:")
        for name, weight in weights.items():
            print(f"{name}: {weight:.4f}")
        
        return weights

    def predict(self, X):
        # Record start time for prediction
        start_time = time.time()
        
        if hasattr(X, 'values'):
            X = X.values

        unique_classes = np.unique(self.y_train)
        n_classes = len(unique_classes)
        max_class = np.max(unique_classes)
        
        predictions = np.zeros((X.shape[0], len(self.models), n_classes))
        prediction_uncertainties = np.zeros((X.shape[0], len(self.models)))
        model_prediction_times = {}
        
        for i, (name, subspace_models) in enumerate(self.trained_models.items()):
            model_start_time = time.time()
            
            subspace_predictions = np.zeros((X.shape[0], len(subspace_models), n_classes))
            
            for j, (model, selected_features) in enumerate(subspace_models):
                X_subspace = X[:, selected_features]
                if hasattr(model, 'predict_proba'):
                    subspace_predictions[:, j, :] = model.predict_proba(X_subspace)
                else:
                    pred = model.predict(X_subspace)
                    pred = pred.astype(int)
                    
                    if np.any(pred > max_class) or np.any(pred < 0):
                        raise ValueError(f"Model {name} produced invalid class predictions. "
                                    f"Expected classes 0 to {max_class}, got {np.unique(pred)}")
                    
                    one_hot = np.zeros((pred.shape[0], n_classes))
                    for idx, p in enumerate(pred):
                        one_hot[idx, p] = 1
                    subspace_predictions[:, j, :] = one_hot
            
            predictions[:, i, :] = np.mean(subspace_predictions, axis=1)
            prediction_uncertainties[:, i] = np.mean(np.var(subspace_predictions, axis=1), axis=1)
            
            # Record prediction time for this model
            model_prediction_time = time.time() - model_start_time
            model_prediction_times[name] = model_prediction_time
            print(f"\n{name} Prediction Time: {model_prediction_time:.2f} seconds")
        
        self.individual_predictions = predictions
        self.prediction_uncertainties = prediction_uncertainties
        
        weighted_predictions = np.sum(predictions * np.array(list(self.model_weights.values()))[:, np.newaxis], axis=1)
        total_uncertainty = np.zeros(X.shape[0])
        
        for i in range(len(self.model_weights)):
            weight = list(self.model_weights.values())[i]
            total_uncertainty += weight * (prediction_uncertainties[:, i] + 
                                        np.mean((predictions[:, i] - weighted_predictions)**2, axis=1))
        
        self.prediction_uncertainty = total_uncertainty
        
        # Record total prediction time
        total_prediction_time = time.time() - start_time
        print(f"\nTotal Prediction Time: {total_prediction_time:.2f} seconds")
        
        self.model_prediction_times = model_prediction_times

        return np.argmax(weighted_predictions, axis=1)

    def evaluate_model(self):
        y_pred = self.predict(self.X_test.values)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        detection_rate = np.sum(y_pred == self.y_test) / len(self.y_test)
        cm = confusion_matrix(self.y_test, y_pred)

        return accuracy, precision, recall, f1, detection_rate, cm

    def evaluate_individual_models(self):
        individual_metrics = {}
        
        for i, model_name in enumerate(self.models.keys()):
            y_pred = np.argmax(self.individual_predictions[:, i, :], axis=1)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            detection_rate = np.sum(y_pred == self.y_test) / len(self.y_test)
            
            individual_metrics[model_name] = {
                'accuracy': accuracy, 
                'precision': precision, 
                'recall': recall, 
                'f1': f1,
                'detection_rate': detection_rate
            }

        return individual_metrics

    def print_results(self):
        accuracy, precision, recall, f1, detection_rate, cm = self.evaluate_model()
        individual_metrics = self.evaluate_individual_models()
        
        print(f"\nEnsemble Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Detection Rate: {detection_rate:.4f}")
        
        self.plot_results(cm, individual_metrics)
    
    def plot_results(self, confusion_mat, individual_metrics):
        # Machine Learning Application Plots
        # 1. Plot confusion matrix
        self.plot_confusion_matrix(confusion_mat)

        # 2. Plot model weights with error bars
        self.plot_model_weights()


        # 3. Individual Models VS Advocated Model Performance
        self.plot_individual_vs_advocated_model(individual_metrics)

        # 4. Plot ROC Curves for individual models and advocated model
        self.plot_roc_curves()

        # 5. Plot prediction uncertainty
        self.plot_prediction_uncertainty()

        # 6. Plot feature importance
        self.plot_feature_importance()

    # Exploratory Data Analysis Plots
    def plot_temporal_analysis(self, df):
        temporal_counts = df[' Timestamp'].value_counts().sort_index()
        print(f"\nTemporal Analysis Results:")
        print(f"Total time periods: {len(temporal_counts)}")
        print(f"Maximum attacks in a period: {temporal_counts.max()}")
        print(f"Average attacks per period: {temporal_counts.mean():.2f}")

        plt.figure(figsize=(15, 6))
        df[' Timestamp'].value_counts().sort_index().plot(kind='line')
        plt.title('Attack Distribution Over Time', pad=20, size=14)
        plt.xlabel('Time', size=12)
        plt.ylabel('Number of Attacks', size=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_protocol_distribution(self, df):
        protocol_counts = df[' Protocol'].value_counts()
        print(f"\nProtocol Distribution Results:")
        print(protocol_counts)

        plt.figure(figsize=(12, 6))
        df[' Protocol'].value_counts().plot(kind='bar', color='skyblue', edgecolor='navy')
        plt.title('Distribution of Protocols', pad=20, size=14)
        plt.xlabel('Protocol', size=12)
        plt.ylabel('Frequency', size=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/protocol_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_correlation(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        print(f"\nCorrelation Matrix:")
        print(correlation_matrix)
        
        plt.figure(figsize=(30, 24))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap', pad=20, size=14)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_flow_duration(self, df):
        flow_duration_stats = df.groupby(' Protocol')[' Flow Duration'].describe()
        print(f"\nFlow Duration Statistics by Protocol:")
        print(flow_duration_stats)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=' Protocol', y=' Flow Duration', data=df)
        plt.title('Flow Duration by Protocol', pad=20, size=14)
        plt.xlabel('Protocol', size=12)
        plt.ylabel('Flow Duration', size=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/flow_duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Statistical Modeling Plots
    def plot_feature_distributions(self, df, data):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:   # Plot first 5 numeric features
            if col != ' Source IP' and col != ' Destination IP':
                feature_stats = df.groupby(' Label')[col].describe()
                print(f"\nDistribution Statistics for {col} by Label:")
                print(feature_stats)

                plt.figure(figsize=(12, 6))
                sns.histplot(data=data, x=col, hue=' Label', multiple="stack")
                plt.title(f'Distribution of {col} by Label', pad=20, size=14)
                plt.xlabel(col, size=12)
                plt.ylabel('Count', size=12)
                plt.tight_layout()
                plt.savefig(f'c:/Xampp/htdocs/net-cure-website/model_plots/distribution_{col.strip()}.png', dpi=300, bbox_inches='tight')
                plt.close()

    def plot_attack_clustering_PCA(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[numeric_cols])

            print(f"\nPCA Analysis Results:")
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            
            plt.figure(figsize=(12, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df[' Label'], 
                    cmap='viridis', alpha=0.6)
            plt.title('Attack Pattern Clustering (PCA)', pad=20, size=14)
            plt.xlabel('First Principal Component', size=12)
            plt.ylabel('Second Principal Component', size=12)
            plt.colorbar(label='Attack Type')
            plt.tight_layout()
            plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/attack_clustering.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Machine Learning Application Plots
    def plot_confusion_matrix(self, confusion_mat):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='YlOrRd',
                    square=True, linewidths=0.5)
        plt.title('Confusion Matrix', pad=20, size=14)
        plt.ylabel('True Label', size=12)
        plt.xlabel('Predicted Label', size=12)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_weights(self):
        plt.figure(figsize=(12, 6))
        weights = list(self.model_weights.values())
        names = list(self.model_weights.keys())
        plt.bar(names, weights, color='skyblue', edgecolor='navy')
        plt.title('Model Weights Distribution', pad=20, size=14)
        plt.xlabel('Models', size=12)
        plt.ylabel('Weight', size=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/model_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_individual_vs_advocated_model(self, individual_metrics):
        # Define the list of metrics you want to plot
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'detection_rate']

        # Get advocated model metrics
        advocated_accuracy, advocated_precision, advocated_recall, advocated_f1, advocated_detection_rate, _ = self.evaluate_model()
        advocated_metrics = {
            'accuracy': advocated_accuracy,
            'precision': advocated_precision,
            'recall': advocated_recall,
            'f1': advocated_f1,
            'detection_rate': advocated_detection_rate
        }

        # Set up the plot
        x_individual = np.arange(len(individual_metrics))
        x_advocated = np.arange(len(individual_metrics), len(individual_metrics) + 1)
        x_combined = np.concatenate([x_individual, x_advocated])
        width = 0.15

        # Plot bars for each metric
        for idx, metric in enumerate(metrics_list):
            # Individual model values
            ind_values = [metrics[metric] for metrics in individual_metrics.values()]
            plt.bar(x_individual + (idx - 2) * width, ind_values, width, 
                    label=f'Individual - {metric.capitalize()}',
                    alpha=0.7)
            
            # Advocated model values (single bar for each metric)
            plt.bar(x_advocated + (idx - 2) * width, [advocated_metrics[metric]], width,
                    label=f'Advocated - {metric.capitalize()}',
                    alpha=0.7)

        plt.xlabel('Models', size=12)
        plt.ylabel('Score', size=12)
        plt.title('Individual vs Advocated Model Performance Metrics', pad=20, size=14)

        # Update x-axis labels to include advocated model
        x_labels = list(individual_metrics.keys()) + ['Advocated Model']
        plt.xticks(x_combined, x_labels, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/individual_vs_advocated_model_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self):
        if hasattr(self, 'individual_predictions'):
            n_classes = self.individual_predictions.shape[2]
            from matplotlib import colormaps
            colors = colormaps['tab10']
            
            # Convert y_test to one-hot encoding
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(self.y_test, classes=range(n_classes))
            
            # Dictionary to store ROC metrics
            roc_metrics = {
                'individual_models': {},
                'advocated_model': {}
            }
            
            print("\nROC Curve Metrics:")
            print("-" * 50)
            
            # Create separate plots for each class
            for class_idx in range(n_classes):
                plt.figure(figsize=(10, 8))
                
                for model_idx, model_name in enumerate(self.models.keys()):
                    if model_name not in roc_metrics['individual_models']:
                        roc_metrics['individual_models'][model_name] = {}
                    
                    y_true = y_test_bin[:, class_idx]
                    y_score = self.individual_predictions[:, model_idx, class_idx]
                    
                    fpr, tpr, thresholds = roc_curve(y_true, y_score)
                    roc_auc = auc(fpr, tpr)
                    
                    # Store metrics
                    roc_metrics['individual_models'][model_name][f'class_{class_idx}'] = {
                        'fpr': fpr,
                        'tpr': tpr,
                        'thresholds': thresholds,
                        'auc': roc_auc
                    }
                    
                    plt.plot(fpr, tpr, 
                            color=colors(model_idx/len(self.models)),
                            linestyle='--', alpha=0.6,
                            label=f'{model_name} (AUC = {roc_auc:.2f})')

                # Plot advocated model for this class
                weighted_predictions = np.zeros((self.individual_predictions.shape[0], n_classes))
                for i, weight in enumerate(self.model_weights.values()):
                    weighted_predictions += weight * self.individual_predictions[:, i, :]

                y_true = y_test_bin[:, class_idx]
                y_score = weighted_predictions[:, class_idx]
                
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Store metrics
                roc_metrics['advocated_model'][f'class_{class_idx}'] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': roc_auc
                }
                
                # Print metrics
                print(f"\nAdvocated Model:")
                print(f"  AUC: {roc_auc:.4f}")
                print(f"  Mean FPR: {np.mean(fpr):.4f}")
                print(f"  Mean TPR: {np.mean(tpr):.4f}")
                
                plt.plot(fpr, tpr,
                        color='red',
                        linewidth=2,
                        label=f'Advocated Model (AUC = {roc_auc:.2f})')

                plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', size=12)
                plt.ylabel('True Positive Rate', size=12)
                plt.title(f'ROC Curves - Class {class_idx}', pad=20, size=14)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'c:/Xampp/htdocs/net-cure-website/model_plots/roc_curves_class_{class_idx}.png', dpi=300, bbox_inches='tight')
                plt.close()

            # Calculate and print macro-averaged AUC for each model
            print("\nMacro-averaged AUC scores:")
            print("-" * 25)
            
            # For individual models
            for model_name in roc_metrics['individual_models']:
                avg_auc = np.mean([metrics['auc'] 
                                for metrics in roc_metrics['individual_models'][model_name].values()])
                print(f"{model_name}: {avg_auc:.4f}")
            
            # For advocated model
            avg_auc_advocated = np.mean([metrics['auc'] 
                                    for metrics in roc_metrics['advocated_model'].values()])
            print(f"Advocated Model: {avg_auc_advocated:.4f}")
            
            # Store ROC metrics for potential later use
            self.roc_metrics = roc_metrics

    def plot_prediction_uncertainty(self):
        if hasattr(self, 'prediction_uncertainty'):
            # Print uncertainty values
            print("\nPrediction Uncertainty Values:")
            print(self.prediction_uncertainty)
            print(f"\nSummary Statistics:")
            print(f"Mean Uncertainty: {np.mean(self.prediction_uncertainty):.3f}")

            # Create the plot
            plt.figure(figsize=(12, 6))
            sns.histplot(data=self.prediction_uncertainty, bins=50, 
                        color='skyblue', edgecolor='navy')
            plt.axvline(x=np.mean(self.prediction_uncertainty), 
                    color='red', linestyle='--', 
                    label=f'Mean Uncertainty: {np.mean(self.prediction_uncertainty):.3f}')
            plt.title('Prediction Uncertainty Distribution', pad=20, size=14)
            plt.xlabel('Uncertainty', size=12)
            plt.ylabel('Frequency', size=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/prediction_uncertainty.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_feature_importance(self):
        if hasattr(self, 'stable_features'):
            # Get feature names from X_train
            feature_names = self.X_train.columns[self.stable_features]
            
            # Create feature importance Series with actual names
            feature_importance = pd.Series(
                self.feature_importance[self.stable_features],
                index=feature_names
            ).sort_values(ascending=False)  # Sort descending for printing
            
            # Print feature importance scores
            print("\nFeature Importance Scores:")
            print("-" * 50)
            for name, score in feature_importance.items():
                print(f"{name:<40} {score:.4f}")
            print("-" * 50)
            
            # Plot
            plt.figure(figsize=(12, 8))
            
            # Sort ascending for plotting (bottom to top)
            feature_importance_plot = feature_importance.sort_values(ascending=True)
            feature_importance_plot.plot(kind='barh', color='skyblue', edgecolor='navy')
            
            plt.title('Feature Importance Scores', pad=20, size=14)
            plt.xlabel('Importance Score', size=12)
            plt.ylabel('Features', size=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig('c:/Xampp/htdocs/net-cure-website/model_plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

# Usage example
if __name__ == "__main__":
    data_paths = [
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_DNS.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_LDAP.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_MSSQL.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_NetBIOS.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_NTP.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_SNMP.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_SSDP.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_UDP.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/Syn.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/TFTP.csv',
        'c:/Xampp/htdocs/net-cure-website/CIC 2019 DDOS DATASET/reduced_10000_rows/UDPLag.csv'
    ]
    # Create an instance of the NovelHybridModel class and print results
    model = NovelHybridModel(data_paths)
    model.print_results()