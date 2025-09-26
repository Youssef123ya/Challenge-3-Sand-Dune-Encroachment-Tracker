"""
Sand Dune Encroachment Tracker - Predictor
Main prediction engine combining all models for sand dune movement forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

from data_processor import SatelliteDataProcessor
from ml_models import CNNModel, LSTMModel, HybridCNNLSTM, TimeSeriesPredictor, ModelEnsemble

class SandDunePredictor:
    """
    Main prediction system for sand dune encroachment tracking
    """

    def __init__(self, config_path: str = 'config.json'):
        """Initialize predictor with configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.data_processor = SatelliteDataProcessor(config_path)
        self.models = {}
        self.ensemble = None
        self.trained = False

        # Initialize prediction parameters
        self.forecast_horizon = self.config['prediction']['forecast_horizon']
        self.confidence_intervals = self.config['prediction']['confidence_intervals']
        self.movement_threshold = self.config['alerts']['movement_threshold']
        self.risk_zones = self.config['alerts']['risk_zones']

    def prepare_training_data(self, image_data: Dict, movement_data: np.ndarray) -> Dict:
        """
        Prepare training data for all models
        """
        print("Preparing training data...")

        # Extract images and preprocess
        images = image_data['images']
        n_samples, height, width = images.shape

        # Prepare CNN data (single images -> movement vectors)
        X_cnn = np.expand_dims(images, axis=-1)  # Add channel dimension
        y_cnn = movement_data

        # Prepare LSTM data (time sequences -> movement vectors)
        sequence_length = self.config['models']['lstm']['sequence_length']

        # Create synthetic time series from images (sliding window approach)
        X_lstm = []
        y_lstm = []

        for i in range(sequence_length, len(images)):
            # Extract spatial features from each image in sequence
            sequence_features = []
            for j in range(i-sequence_length, i):
                # Calculate spatial statistics as features
                img = images[j]
                features = [
                    np.mean(img),
                    np.std(img),
                    np.max(img),
                    np.min(img),
                    np.median(img),
                    np.percentile(img, 25),
                    np.percentile(img, 75),
                    np.sum(img > np.mean(img))  # Above-average pixels
                ]
                sequence_features.append(features)

            X_lstm.append(sequence_features)
            y_lstm.append(movement_data[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        # Prepare Hybrid CNN-LSTM data (spatiotemporal sequences)
        X_hybrid = []
        y_hybrid = []

        timesteps = 5  # Number of timesteps in sequence
        for i in range(timesteps, len(images)):
            sequence = images[i-timesteps:i]
            X_hybrid.append(np.expand_dims(sequence, axis=-1))
            y_hybrid.append(movement_data[i])

        X_hybrid = np.array(X_hybrid)
        y_hybrid = np.array(y_hybrid)

        # Prepare traditional ML data (spatial features -> movement)
        X_traditional = []
        y_traditional = movement_data

        for img in images:
            # Extract comprehensive spatial features
            features = [
                # Basic statistics
                np.mean(img), np.std(img), np.max(img), np.min(img),
                np.median(img), np.var(img),

                # Percentiles
                np.percentile(img, 10), np.percentile(img, 25),
                np.percentile(img, 75), np.percentile(img, 90),

                # Texture features
                np.mean(np.gradient(img)[0]), np.mean(np.gradient(img)[1]),
                np.std(np.gradient(img)[0]), np.std(np.gradient(img)[1]),

                # Shape features
                np.sum(img > np.mean(img)) / img.size,  # Above-mean ratio
                np.sum(img > np.percentile(img, 75)) / img.size,  # High-value ratio

                # Spatial moments
                np.sum(img * np.arange(img.shape[0])[:, None]) / np.sum(img),  # Centroid Y
                np.sum(img * np.arange(img.shape[1])[None, :]) / np.sum(img),  # Centroid X
            ]
            X_traditional.append(features)

        X_traditional = np.array(X_traditional)

        return {
            'cnn': {'X': X_cnn, 'y': y_cnn},
            'lstm': {'X': X_lstm, 'y': y_lstm},
            'hybrid': {'X': X_hybrid, 'y': y_hybrid},
            'traditional': {'X': X_traditional, 'y': y_traditional}
        }

    def train_all_models(self, training_data: Dict) -> Dict:
        """
        Train all models with prepared data
        """
        print("Training all models...")
        training_results = {}

        # Train CNN Model
        print("\n1. Training CNN Model...")
        cnn_model = CNNModel(self.config['models']['cnn'])

        X_cnn, y_cnn = training_data['cnn']['X'], training_data['cnn']['y']

        # Split data
        split_idx = int(0.8 * len(X_cnn))
        X_train, X_val = X_cnn[:split_idx], X_cnn[split_idx:]
        y_train, y_val = y_cnn[:split_idx], y_cnn[split_idx:]

        cnn_model.build_model(X_cnn.shape[1:])
        cnn_history = cnn_model.train(X_train, y_train, X_val, y_val)

        self.models['cnn'] = cnn_model
        training_results['cnn'] = cnn_history

        # Train LSTM Model
        print("\n2. Training LSTM Model...")
        lstm_model = LSTMModel(self.config['models']['lstm'])

        X_lstm, y_lstm = training_data['lstm']['X'], training_data['lstm']['y']

        if len(X_lstm) > 0:  # Check if we have sequence data
            split_idx = int(0.8 * len(X_lstm))
            X_train, X_val = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train, y_val = y_lstm[:split_idx], y_lstm[split_idx:]

            lstm_model.build_model(X_lstm.shape[1:])
            lstm_history = lstm_model.train(X_train, y_train, X_val, y_val)

            self.models['lstm'] = lstm_model
            training_results['lstm'] = lstm_history

        # Train Hybrid CNN-LSTM Model
        print("\n3. Training Hybrid CNN-LSTM Model...")
        hybrid_model = HybridCNNLSTM(self.config['models']['hybrid_cnn_lstm'])

        X_hybrid, y_hybrid = training_data['hybrid']['X'], training_data['hybrid']['y']

        if len(X_hybrid) > 0:  # Check if we have spatiotemporal data
            split_idx = int(0.8 * len(X_hybrid))
            X_train, X_val = X_hybrid[:split_idx], X_hybrid[split_idx:]
            y_train, y_val = y_hybrid[:split_idx], y_hybrid[split_idx:]

            hybrid_model.build_model(X_hybrid.shape[1:])
            hybrid_history = hybrid_model.train(X_train, y_train, X_val, y_val)

            self.models['hybrid'] = hybrid_model
            training_results['hybrid'] = hybrid_history

        # Train Traditional Models
        print("\n4. Training Traditional ML Models...")
        ts_predictor = TimeSeriesPredictor()

        X_trad, y_trad = training_data['traditional']['X'], training_data['traditional']['y']

        # Train Random Forest (for combined x,y movement)
        y_magnitude = np.linalg.norm(y_trad, axis=1)  # Movement magnitude
        rf_results = ts_predictor.fit_random_forest(X_trad, y_magnitude)

        # Train SVM
        svm_results = ts_predictor.fit_svm(X_trad, y_magnitude)

        self.models['traditional'] = ts_predictor
        training_results['traditional'] = {
            'random_forest': rf_results['metrics'],
            'svm': svm_results['metrics']
        }

        # Create Ensemble Model
        print("\n5. Creating Ensemble Model...")
        available_models = [model for model in self.models.values() if hasattr(model, 'predict')]

        if len(available_models) > 1:
            # Equal weights for now, could be optimized based on validation performance
            weights = [1.0/len(available_models)] * len(available_models)
            self.ensemble = ModelEnsemble(available_models, weights)

        self.trained = True
        print("\nAll models trained successfully!")

        return training_results

    def predict_movement(self, input_data: np.ndarray, method: str = 'ensemble') -> Dict:
        """
        Predict sand dune movement
        """
        if not self.trained:
            raise ValueError("Models must be trained before making predictions")

        predictions = {}

        if method == 'cnn' and 'cnn' in self.models:
            pred = self.models['cnn'].predict(input_data)
            predictions['cnn'] = {
                'movement': pred,
                'confidence': None
            }

        elif method == 'lstm' and 'lstm' in self.models:
            pred = self.models['lstm'].predict(input_data)
            predictions['lstm'] = {
                'movement': pred,
                'confidence': None
            }

        elif method == 'hybrid' and 'hybrid' in self.models:
            pred = self.models['hybrid'].predict(input_data)
            predictions['hybrid'] = {
                'movement': pred,
                'confidence': None
            }

        elif method == 'ensemble' and self.ensemble is not None:
            # For ensemble, we need to adapt input for different models
            pred_mean, pred_std = self.ensemble.predict_with_uncertainty(input_data)
            predictions['ensemble'] = {
                'movement': pred_mean,
                'confidence': pred_std,
                'confidence_intervals': self._calculate_confidence_intervals(pred_mean, pred_std)
            }

        return predictions

    def forecast_trajectory(self, initial_position: np.ndarray, n_steps: int, 
                          input_data: np.ndarray, method: str = 'ensemble') -> Dict:
        """
        Forecast sand dune trajectory over multiple time steps
        """
        trajectory = [initial_position.copy()]
        movements = []
        confidences = []

        current_data = input_data.copy()

        for step in range(n_steps):
            # Predict next movement
            prediction = self.predict_movement(current_data, method)

            if method in prediction:
                movement = prediction[method]['movement']
                confidence = prediction[method].get('confidence', None)

                # Update position
                if len(movement.shape) == 1:
                    new_position = trajectory[-1] + movement
                else:
                    new_position = trajectory[-1] + movement[0]  # Take first prediction if batch

                trajectory.append(new_position)
                movements.append(movement)

                if confidence is not None:
                    confidences.append(confidence)

                # Update input data for next prediction (simplified approach)
                # In practice, this would involve more sophisticated data updating
                current_data = self._update_input_data(current_data, movement)

        return {
            'trajectory': np.array(trajectory),
            'movements': np.array(movements),
            'confidences': np.array(confidences) if confidences else None,
            'total_displacement': trajectory[-1] - trajectory[0],
            'max_displacement': np.max([np.linalg.norm(pos - trajectory[0]) for pos in trajectory])
        }

    def assess_risk(self, trajectory: np.ndarray, infrastructure_locations: List[np.ndarray]) -> Dict:
        """
        Assess risk to infrastructure based on predicted trajectory
        """
        risks = []

        for i, infra_pos in enumerate(infrastructure_locations):
            min_distance = float('inf')
            closest_time = 0

            for t, dune_pos in enumerate(trajectory):
                distance = np.linalg.norm(dune_pos - infra_pos)
                if distance < min_distance:
                    min_distance = distance
                    closest_time = t

            # Determine risk level
            if min_distance <= self.risk_zones['high']:
                risk_level = 'HIGH'
            elif min_distance <= self.risk_zones['medium']:
                risk_level = 'MEDIUM'
            elif min_distance <= self.risk_zones['low']:
                risk_level = 'LOW'
            else:
                risk_level = 'MINIMAL'

            risks.append({
                'infrastructure_id': i,
                'risk_level': risk_level,
                'min_distance': min_distance,
                'closest_time_step': closest_time,
                'estimated_days': closest_time * 30  # Assuming monthly predictions
            })

        return {
            'risks': risks,
            'high_risk_count': sum(1 for r in risks if r['risk_level'] == 'HIGH'),
            'overall_risk': max([r['risk_level'] for r in risks], 
                              key=lambda x: ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH'].index(x))
        }

    def generate_alerts(self, risk_assessment: Dict, movement_prediction: Dict) -> List[Dict]:
        """
        Generate alerts based on risk assessment and movement predictions
        """
        alerts = []

        # Movement-based alerts
        if 'ensemble' in movement_prediction:
            movement = movement_prediction['ensemble']['movement']
            if len(movement.shape) > 1:
                movement = movement[0]  # Take first prediction

            movement_magnitude = np.linalg.norm(movement)

            if movement_magnitude > self.movement_threshold:
                alerts.append({
                    'type': 'MOVEMENT_ALERT',
                    'severity': 'WARNING',
                    'message': f'Significant sand movement detected: {movement_magnitude:.2f} meters',
                    'movement_vector': movement.tolist(),
                    'timestamp': datetime.now().isoformat()
                })

        # Infrastructure risk alerts
        for risk in risk_assessment['risks']:
            if risk['risk_level'] in ['HIGH', 'MEDIUM']:
                alerts.append({
                    'type': 'INFRASTRUCTURE_RISK',
                    'severity': risk['risk_level'],
                    'message': f'Infrastructure {risk["infrastructure_id"]} at risk in {risk["estimated_days"]} days',
                    'distance': risk['min_distance'],
                    'estimated_days': risk['estimated_days'],
                    'timestamp': datetime.now().isoformat()
                })

        return alerts

    def _calculate_confidence_intervals(self, mean: np.ndarray, std: np.ndarray) -> Dict:
        """Calculate confidence intervals"""
        intervals = {}

        for ci in self.confidence_intervals:
            z_score = {0.68: 1.0, 0.95: 1.96, 0.99: 2.58}.get(ci, 1.96)
            intervals[f'{int(ci*100)}%'] = {
                'lower': mean - z_score * std,
                'upper': mean + z_score * std
            }

        return intervals

    def _update_input_data(self, current_data: np.ndarray, movement: np.ndarray) -> np.ndarray:
        """Update input data based on predicted movement (simplified)"""
        # This is a simplified update - in practice would involve more sophisticated 
        # simulation of how the image changes based on movement
        return current_data

    def save_models(self, save_dir: str):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.models.items():
            if hasattr(model, 'model') and model.model is not None:
                model.model.save(os.path.join(save_dir, f'{name}_model.h5'))

                # Save scaler if exists
                if hasattr(model, 'scaler'):
                    with open(os.path.join(save_dir, f'{name}_scaler.pkl'), 'wb') as f:
                        pickle.dump(model.scaler, f)

        # Save ensemble if exists
        if self.ensemble is not None:
            with open(os.path.join(save_dir, 'ensemble.pkl'), 'wb') as f:
                pickle.dump(self.ensemble, f)

        print(f"Models saved to {save_dir}")

    def load_models(self, load_dir: str):
        """Load trained models"""
        # Implementation would load saved models
        # This is a placeholder for the full implementation
        print(f"Loading models from {load_dir}")
        self.trained = True

# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = SandDunePredictor()

    # Create synthetic data for testing
    synthetic_data = predictor.data_processor.create_synthetic_data(n_samples=500)

    # Prepare training data
    training_data = predictor.prepare_training_data(synthetic_data, synthetic_data['movements'])

    print("Training data prepared:")
    for key, data in training_data.items():
        print(f"  {key}: X shape {data['X'].shape}, y shape {data['y'].shape}")

    # Train models (with smaller dataset for testing)
    # training_results = predictor.train_all_models(training_data)

    print("\nSand Dune Predictor system ready!")
    print("Features:")
    print("- Multi-model ensemble prediction")
    print("- Trajectory forecasting")
    print("- Risk assessment for infrastructure")
    print("- Automated alert generation")
