class Performance {
    constructor() {
        this.message = null;
    }

    initializePlots() {
        const plots = [
            { id: 'confusion-matrix', filename: '../../model_plots/confusion_matrix.png' },
            { id: 'individual-vs-advocated-model-performance-metrics', filename: '../../model_plots/individual_vs_advocated_model_performance_metrics.png' },
            { id: 'model-weights', filename: '../../model_plots/model_weights.png' },
            { id: 'prediction-uncertainty', filename: '../../model_plots/prediction_uncertainty.png' },
            { id: 'feature-importance', filename: '../../model_plots/feature_importance.png' },
            { id: 'temporal-analysis', filename: '../../model_plots/temporal_analysis.png' },
            { id: 'correlation-heatmap', filename: '../../model_plots/correlation_heatmap.png' },
            { id: 'flow-duration-analysis', filename: '../../model_plots/flow_duration_analysis.png' },
            { id: 'attack-clustering', filename: '../../model_plots/attack_clustering.png' },
            { id: 'protocol-distribution', filename: '../../model_plots/protocol_distribution.png' },
            { id: 'distribution-source-port', filename: '../../model_plots/distribution_Source Port.png' },
            { id: 'distribution-destination-port', filename: '../../model_plots/distribution_Destination Port.png' },
            { id: 'distribution-protocol', filename: '../../model_plots/distribution_Protocol.png' },
            { id: 'roc-curves-class-0', filename: '../../model_plots/roc_curves_class_0.png' },
            { id: 'roc-curves-class-1', filename: '../../model_plots/roc_curves_class_1.png' },
            { id: 'roc-curves-class-2', filename: '../../model_plots/roc_curves_class_2.png' },
            { id: 'roc-curves-class-3', filename: '../../model_plots/roc_curves_class_3.png' },
            { id: 'roc-curves-class-4', filename: '../../model_plots/roc_curves_class_4.png' },
            { id: 'roc-curves-class-5', filename: '../../model_plots/roc_curves_class_5.png' },
            { id: 'roc-curves-class-6', filename: '../../model_plots/roc_curves_class_6.png' },
            { id: 'roc-curves-class-7', filename: '../../model_plots/roc_curves_class_7.png' },
            { id: 'roc-curves-class-8', filename: '../../model_plots/roc_curves_class_8.png' },
            { id: 'roc-curves-class-9', filename: '../../model_plots/roc_curves_class_9.png' },
            { id: 'roc-curves-class-10', filename: '../../model_plots/roc_curves_class_10.png' },
            { id: 'roc-curves-class-11', filename: '../../model_plots/roc_curves_class_11.png' },
        ];
    
        plots.forEach(plot => {
            const img = document.getElementById(plot.id);
            if (img) {
                img.src = plot.filename;
                img.onerror = function() {
                    this.src = '../../model_plots/placeholder.png';
                    this.alt = 'No Image Found';
                };
            }
        });
    }

    getPerformance() {                
        this.message = "Model Performance Loaded Successfully!";
        $('#responseMessage').html(this.message);
        this.initializePlots();
    }
}

$(document).ready(function() {
    const performance = new Performance();
    performance.getPerformance();
});