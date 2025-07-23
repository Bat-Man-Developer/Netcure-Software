<?php
    // Store Information
    $productimagename = "confusion_matrix";
    $productimagename1 = "feature_importance";
    $productimagename2 = "model_weights";
    $productimagename3 = "performance_metrics";
    $productimagename4 = "roc_curve";
    $productimagename5 = "precision_recall_curve";
    $productimagename6 = "calibration_plot";
    $productimagename7 = "boxplots_predictions";
    $productimagename8 = "accuracy_comaprison";
    $productimagename9 = "precision_comaprison";
    $productimagename10 = "recall_comaprison";
    $productimagename11 = "f1_comaprison";
    $productimagename12 = "bias_variance_tradeoff";
    $productimagename13 = "learning_curves";

    // Image Path
    $srcFilePath = $_SERVER['DOCUMENT_ROOT']."model_plots/";
    $specificFiles = array(
        $srcFilePath.$productimagename,
        $srcFilePath.$productimagename1,
        $srcFilePath.$productimagename2,
        $srcFilePath.$productimagename3,
        $srcFilePath.$productimagename4,
        $srcFilePath.$productimagename5,
        $srcFilePath.$productimagename6,
        $srcFilePath.$productimagename7,
        $srcFilePath.$productimagename8,
        $srcFilePath.$productimagename9,
        $srcFilePath.$productimagename10,
        $srcFilePath.$productimagename11,
        $srcFilePath.$productimagename12,
        $srcFilePath.$productimagename13,
    );

    // Delete Image files in Directory
    foreach($specificFiles as $file){
        if(file_exists($file)){
            unlink($file);  
        }
    }