#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Training Utilities
"""

import os
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt


def train_model(model, dataset, train_dataset, validation_dataset, train_config=None):
    """
    Train the model and save it
    
    Parameters:
    - model: Model to be trained
    - dataset: Complete dataset (for debugging)
    - train_dataset: Training dataset
    - validation_dataset: Validation dataset
    - train_config: Training configuration parameters (optional)
    
    Returns:
    - history: Training history
    """
    # Get training parameters
    if train_config is None:
        # Default values
        epochs = 2
        batch_size = 256
        lr = 0.0005
        weight_decay = 0.001
    else:
        epochs = train_config['training']['epochs'] 
        batch_size = train_config['training']['batch_size']
        lr = train_config['training']['lr']
        weight_decay = train_config['training']['weight_decay']
    
    # Define callbacks
    callbacks = [
        # Model checkpoint - using SavedModel format instead of HDF5
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./models/push_binary_classification_model",
            save_best_only=False,
            save_weights_only=False,
            save_format="tf"),  # 使用SavedModel格式
        # Save best model - using SavedModel format
        tf.keras.callbacks.ModelCheckpoint(
            filepath="./models/best_model",
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            save_weights_only=False,
            save_format="tf"),  # 使用SavedModel格式
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=2,
            mode='max',
            restore_best_weights=True),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=0.00001,
            verbose=1),
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            f'./logs/training_log_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            append=False)
    ]
    
    # Create log directory
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    
    # Record training start time
    start_time = time.time()
    
    # Train model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Record training end time
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed, total time: {training_time:.2f} seconds")
    
    # 在训练完成后保存最终模型
    try:
        # 使用标准保存方法，避免使用实验性选项
        model.save("./models/final_model", save_format="tf")
        print("模型已保存到 ./models/final_model")
    except Exception as e:
        print(f"保存最终模型时出错: {str(e)}")
    
    # Get final training and validation metrics
    final_train_auc = history.history['auc'][-1]
    final_val_auc = history.history['val_auc'][-1]
    
    print(f"Final training AUC: {final_train_auc:.4f}")
    print(f"Final validation AUC: {final_val_auc:.4f}")
    print(f"AUC gap: {abs(final_train_auc - final_val_auc):.4f}")
    
    # Plot AUC curve during training
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Training and Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./logs/auc_curve_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    return history 