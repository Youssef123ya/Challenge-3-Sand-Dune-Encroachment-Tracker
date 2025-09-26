"""
Sand Dune Encroachment Tracker - Machine Learning Models
CNN, LSTM, and Hybrid CNN-LSTM models for sand dune movement prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CNNModel:
    """
    Convolutional Neural Network for spatial feature extraction from satellite imagery
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def build_model(self, input_shape: Tuple[int, int, int]) -> keras.Model:
        """Build CNN architecture"""
        model = models.Sequential([
            layers.Input(shape=input_shape),

            # First CNN block
            layers.Conv2D(self.config['filters'][0], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(self.config['filters'][0], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(self.config['dropout']),

            # Second CNN block
            layers.Conv2D(self.config['filters'][1], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(self.config['filters'][1], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(self.config['dropout']),

            # Third CNN block
            layers.Conv2D(self.config['filters'][2], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(self.config['filters'][2], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(self.config['dropout']),

            # Fourth CNN block
            layers.Conv2D(self.config['filters'][3], self.config['kernel_size'], 
                         activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),

            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.Dropout(self.config['dropout']),
            layers.Dense(256, activation='relu'),
            layers.Dropout(self.config['dropout']),
            layers.Dense(2, activation='linear')  # Output: [x_movement, y_movement]
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train the CNN model"""

        # Normalize data
        X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)

        # Callbacks
        early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=5)

        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        return self.model.predict(X_scaled)

class LSTMModel:
    """
    LSTM Neural Network for temporal sequence modeling
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM architecture"""
        model = models.Sequential([
            layers.Input(shape=input_shape),

            # First LSTM layer
            layers.LSTM(self.config['units'][0], return_sequences=True, dropout=self.config['dropout']),
            layers.BatchNormalization(),

            # Second LSTM layer
            layers.LSTM(self.config['units'][1], return_sequences=True, dropout=self.config['dropout']),
            layers.BatchNormalization(),

            # Third LSTM layer
            layers.LSTM(self.config['units'][2], return_sequences=False, dropout=self.config['dropout']),

            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.config['dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dropout(self.config['dropout']),
            layers.Dense(2, activation='linear')  # Output: [x_movement, y_movement]
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM"""
        sequence_length = self.config['sequence_length']
        X, y = [], []

        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train the LSTM model"""

        # Normalize data
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)

        # Callbacks
        early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=5)

        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)
        return self.model.predict(X_scaled)

class HybridCNNLSTM:
    """
    Hybrid CNN-LSTM model combining spatial and temporal analysis
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def build_model(self, input_shape: Tuple[int, int, int, int]) -> keras.Model:
        """Build Hybrid CNN-LSTM architecture"""
        # Input: (batch, timesteps, height, width, channels)

        # CNN feature extraction for each timestep
        cnn_input = layers.Input(shape=input_shape[1:])  # (height, width, channels)

        # CNN layers
        x = layers.Conv2D(self.config['cnn_filters'][0], 3, activation='relu', padding='same')(cnn_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(self.config['cnn_filters'][1], 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)

        x = layers.Conv2D(self.config['cnn_filters'][2], 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        # Create CNN model for feature extraction
        cnn_model = models.Model(cnn_input, x)

        # Main model with TimeDistributed CNN + LSTM
        main_input = layers.Input(shape=input_shape)

        # Apply CNN to each timestep
        cnn_features = layers.TimeDistributed(cnn_model)(main_input)

        # LSTM layers
        lstm_out = layers.LSTM(self.config['lstm_units'][0], return_sequences=True, 
                              dropout=self.config['dropout'])(cnn_features)
        lstm_out = layers.LSTM(self.config['lstm_units'][1], return_sequences=False, 
                              dropout=self.config['dropout'])(lstm_out)

        # Dense layers
        dense_out = layers.Dense(256, activation='relu')(lstm_out)
        dense_out = layers.Dropout(self.config['dropout'])(dense_out)
        dense_out = layers.Dense(128, activation='relu')(dense_out)
        dense_out = layers.Dropout(self.config['dropout'])(dense_out)
        output = layers.Dense(2, activation='linear')(dense_out)  # [x_movement, y_movement]

        model = models.Model(main_input, output)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Train the Hybrid model"""

        # Normalize data
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, original_shape[-1])
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(original_shape)

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.scaler.transform(X_val_flat)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            validation_data = (X_val_scaled, y_val)

        # Callbacks
        early_stopping = callbacks.EarlyStopping(patience=15, restore_best_weights=True)
        reduce_lr = callbacks.ReduceLROnPlateau(factor=0.2, patience=7)

        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        original_shape = X.shape
        X_flat = X.reshape(-1, original_shape[-1])
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(original_shape)
        return self.model.predict(X_scaled)

class TimeSeriesPredictor:
    """
    Traditional time series analysis using ARIMA and other statistical methods
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}

    def fit_random_forest(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Fit Random Forest model"""
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.models['random_forest'] = rf_model

        return {
            'model': rf_model,
            'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test
        }

    def fit_svm(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Fit Support Vector Machine"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit SVM
        svm_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        svm_model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = svm_model.predict(X_test_scaled)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        self.models['svm'] = svm_model
        self.scalers['svm'] = scaler

        return {
            'model': svm_model,
            'scaler': scaler,
            'metrics': {'mse': mse, 'mae': mae, 'r2': r2},
            'predictions': y_pred,
            'actual': y_test
        }

class ModelEnsemble:
    """
    Ensemble of multiple models for improved prediction accuracy
    """

    def __init__(self, models: List, weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0/len(models)] * len(models)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates"""
        predictions = []

        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate mean and standard deviation
        mean_pred = np.average(predictions, axis=0, weights=self.weights)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

# Example usage and testing
if __name__ == "__main__":
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Create synthetic data for testing
    n_samples = 1000
    img_height, img_width = 64, 64  # Smaller size for testing

    # Generate synthetic image data
    X_images = np.random.rand(n_samples, img_height, img_width, 3)
    y_movements = np.random.randn(n_samples, 2) * 5  # Random movements

    # Test CNN model
    print("Testing CNN Model...")
    cnn_model = CNNModel(config['models']['cnn'])
    cnn_model.build_model((img_height, img_width, 3))

    print(f"CNN Model created with {cnn_model.model.count_params()} parameters")

    # Test LSTM model with sequence data
    print("\nTesting LSTM Model...")
    lstm_model = LSTMModel(config['models']['lstm'])

    # Create sequence data
    sequence_length = 10
    n_features = 20
    X_sequences = np.random.rand(n_samples, sequence_length, n_features)

    lstm_model.build_model((sequence_length, n_features))
    print(f"LSTM Model created with {lstm_model.model.count_params()} parameters")

    # Test Hybrid CNN-LSTM
    print("\nTesting Hybrid CNN-LSTM Model...")
    hybrid_model = HybridCNNLSTM(config['models']['hybrid_cnn_lstm'])

    # Create spatiotemporal data
    X_spatiotemporal = np.random.rand(n_samples, 5, 32, 32, 1)  # (batch, time, height, width, channels)

    hybrid_model.build_model((5, 32, 32, 1))
    print(f"Hybrid Model created with {hybrid_model.model.count_params()} parameters")

    print("\nAll models created successfully!")
