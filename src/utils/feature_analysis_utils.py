#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature Analysis Utilities
"""

import json
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def check_feature_importance(model, dataset, train_config=None):
    """
    Check feature importance using permutation importance method
    
    Parameters:
    - model: Trained model
    - dataset: Test dataset
    - train_config: Training configuration
    
    Returns:
    - feature_importance: Feature importance dictionary
    """
    # Set parameters
    num_batches = 5
    
    # Use sampled data for evaluation
    print("Evaluating feature importance using sampled data...")
    print(f"Using {num_batches} batches for evaluation")
    sample_dataset = dataset.take(num_batches)
    
    # Get original performance as baseline
    all_labels = []
    all_preds = []
    
    # Collect predictions and actual labels
    print("Collecting baseline predictions...")
    for i, (x, y) in enumerate(sample_dataset):
        try:
            # Make predictions using the model
            preds = model(x)
            # Ensure predictions and labels are 1D arrays
            y_pred = preds.numpy().flatten()
            y_true = y.numpy().flatten()
            
            all_preds.extend(y_pred)
            all_labels.extend(y_true)
            print(f"Processed batch {i+1}/{num_batches}")
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
    
    # Calculate baseline AUC
    baseline_auc = roc_auc_score(all_labels, all_preds)
    print(f"Baseline AUC: {baseline_auc:.4f}")
    
    # Get all feature names
    first_batch = next(iter(sample_dataset))
    feature_names = list(first_batch[0].keys())
    
    # Evaluate importance for each feature
    feature_importance = {}
    
    for feature_name in feature_names:
        print(f"Evaluating importance of feature '{feature_name}'...")
        feature_aucs = []
        
        # Perform multiple repetitions to increase stability
        for repeat in range(3):
            # Use feature replacement method
            all_preds_permuted = []
            all_labels_permuted = []
            
            # Iterate over the dataset
            for i, (x, y) in enumerate(sample_dataset):
                try:
                    # Create a copy of x
                    x_copy = {k: tf.identity(v) for k, v in x.items()}
                    
                    # Randomize the specific feature
                    if x_copy[feature_name].shape[0] > 1:  # Ensure batch has multiple samples
                        x_copy[feature_name] = tf.random.shuffle(x_copy[feature_name])
                    
                    # Make predictions
                    preds = model(x_copy)
                    all_preds_permuted.extend(preds.numpy().flatten())
                    all_labels_permuted.extend(y.numpy().flatten())
                except Exception as e:
                    print(f"  Prediction error: {e}")
                    continue
            
            # If enough predictions were collected
            if len(all_preds_permuted) > 0:
                # Calculate AUC after feature randomization
                permuted_auc = roc_auc_score(all_labels_permuted, all_preds_permuted)
                feature_aucs.append(permuted_auc)
        
        # Calculate average AUC
        if feature_aucs:
            avg_permuted_auc = np.mean(feature_aucs)
            # Feature importance = baseline AUC - AUC after randomization
            importance = baseline_auc - avg_permuted_auc
            feature_importance[feature_name] = importance
            print(f"  Feature {feature_name} importance: {importance:.6f}")
        else:
            print(f"  Unable to evaluate importance of feature {feature_name}")
            feature_importance[feature_name] = 0.0
    
    # Sort by importance
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    # Print feature importance
    print("\nFeature Importance Ranking:")
    for i, (feature, importance) in enumerate(sorted_importance.items()):
        print(f"{i+1}. {feature}: {importance:.6f}")
    
    # Save feature importance to file
    importance_file = f"./logs/feature_importance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(importance_file, 'w') as f:
        json.dump(sorted_importance, f, indent=2)
    
    print(f"Feature importance saved to: {importance_file}")
    
    return sorted_importance


def plot_feature_importance(feature_importance):
    """Plot feature importance chart"""
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_features]
    importance = [x[1] for x in sorted_features]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importance, color='skyblue')
    plt.xlabel('Importance Score (Baseline AUC - Permuted AUC)')
    plt.ylabel('Feature')
    plt.title('Feature Importance Ranking')
    plt.tight_layout()
    plt.savefig(f'./logs/feature_importance_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    print("Feature importance chart saved") 