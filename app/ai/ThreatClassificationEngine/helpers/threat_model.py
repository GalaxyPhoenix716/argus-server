"""
Threat Classification Model

Gradient Boosted Trees classifier for threat classification.
Optimized for class imbalance and real-time inference.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None

try:
    import optuna
except ImportError:
    optuna = None

from .feature_extractor import AllFeatures
from app.ai.ThreatClassificationEngine.config.threat_config import ModelConfig, TrainingConfig, get_config

logger = logging.getLogger(__name__)


class ThreatClassificationModel:
    """
    Gradient Boosted Trees model for threat classification.

    Handles class imbalance through SMOTE and focal loss.
    Optimized for real-time inference with ONNX export support.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize threat classification model.

        Args:
            model_config: Model configuration (uses default if not provided)
            training_config: Training configuration (uses default if not provided)
        """
        self.config = model_config or get_config().model
        self.training_config = training_config or get_config().training

        # Initialize model
        self._initialize_model()

        # State
        self.feature_names = None
        self.fitted = False
        self.class_counts = None
        self.feature_importances_ = None

    def _initialize_model(self):
        """Initialize the underlying classifier"""
        self.model = HistGradientBoostingClassifier(
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            early_stopping=self.config.early_stopping,
            validation_fraction=self.training_config.validation_fraction,
            n_iter_no_change=self.config.n_iter_no_change,
            class_weight=self.config.class_weight,
            random_state=self.training_config.random_state
        )

        # Initialize SMOTE if enabled
        if self.training_config.use_smote:
            self.smote = SMOTE(
                sampling_strategy=self.training_config.smote_sampling_strategy,
                k_neighbors=self.training_config.smote_neighbors,
                random_state=self.training_config.random_state
            )
        else:
            self.smote = None

        # Sample weighting for focal loss
        if self.config.sample_weight == "focal_loss":
            self.use_focal_weight = True
        else:
            self.use_focal_weight = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None
    ) -> "ThreatClassificationModel":
        """
        Fit the model to training data.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            feature_names: Optional list of feature names

        Returns:
            self
        """
        logger.info(f"Training model on {X.shape[0]} samples with {X.shape[1]} features")

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Validate inputs
        X, y = self._validate_inputs(X, y)

        # Split data
        if self.training_config.temporal_split:
            # Temporal split: use last portion for validation/test
            split_idx = int(X.shape[0] * self.training_config.train_split)
            X_train, X_temp = X[:split_idx], X[split_idx:]
            y_train, y_temp = y[:split_idx], y[split_idx:]

            val_split = self.training_config.val_split / (self.training_config.val_split + self.training_config.test_split)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_split),
                random_state=self.training_config.random_state,
                stratify=y_temp
            )
        else:
            # Random split
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=(self.training_config.val_split + self.training_config.test_split),
                random_state=self.training_config.random_state,
                stratify=y
            )

            val_split = self.training_config.val_split / (self.training_config.val_split + self.training_config.test_split)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_split),
                random_state=self.training_config.random_state,
                stratify=y_temp
            )

        # Apply SMOTE if enabled
        if self.smote is not None:
            logger.info("Applying SMOTE for class balancing")
            X_train, y_train = self.smote.fit_resample(X_train, y_train)

        # Compute sample weights
        sample_weights = None
        if self.use_focal_weight:
            logger.info("Applying focal loss sample weights")
            sample_weights = self._compute_focal_weights(y_train)

        # Hyperparameter tuning if enabled
        if self.training_config.use_optuna:
            logger.info("Starting hyperparameter optimization with Optuna")
            best_params = self._optimize_hyperparameters(X_train, y_train, sample_weights)
            self.model.set_params(**best_params)

        # Train model
        logger.info("Training final model")
        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Store class counts
        self.class_counts = np.bincount(y)

        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Evaluate on validation set
        if hasattr(self.model, 'score'):
            val_score = self.model.score(X_val, y_val)
            logger.info(f"Validation accuracy: {val_score:.4f}")

        # Store test predictions for final evaluation
        self.test_predictions = self.model.predict(X_test)
        self.test_probabilities = self.model.predict_proba(X_test)

        # Print classification report
        logger.info("\nClassification Report:\n" + classification_report(y_test, self.test_predictions))

        self.fitted = True
        logger.info("Model training completed")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict threat classes.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted class labels (n_samples,)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_inputs(X)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_inputs(X)

        return self.model.predict_proba(X)

    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with confidence scores.

        Args:
            X: Features (n_samples, n_features)
            confidence_threshold: Minimum confidence for classification

        Returns:
            Tuple of (predictions, probabilities, confidences)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X = self._validate_inputs(X)

        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        # Apply confidence threshold (set to unknown class if below threshold)
        predictions[confidences < confidence_threshold] = -1  # Unknown class

        return predictions, probabilities, confidences

    def get_feature_importance(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            top_k: Return top K features (None for all)

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.fitted:
            raise ValueError("Model must be fitted to get feature importances")

        importances = self.feature_importances_

        if top_k is not None:
            # Get indices of top features
            top_indices = np.argsort(importances)[-top_k:][::-1]
            importances = importances[top_indices]

            if self.feature_names is not None:
                feature_names = [self.feature_names[i] for i in top_indices]
            else:
                feature_names = [f"feature_{i}" for i in top_indices]
        else:
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        return dict(zip(feature_names, importances))

    def save_model(self, filepath: str):
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'class_counts': self.class_counts,
            'feature_importances_': self.feature_importances_,
            'config': self.config,
            'training_config': self.training_config,
            'fitted': True
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "ThreatClassificationModel":
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded ThreatClassificationModel instance
        """
        model_data = joblib.load(filepath)

        instance = cls(
            model_config=model_data['config'],
            training_config=model_data['training_config']
        )

        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.class_counts = model_data['class_counts']
        instance.feature_importances_ = model_data['feature_importances_']
        instance.fitted = model_data['fitted']

        logger.info(f"Model loaded from {filepath}")

        return instance

    def _validate_inputs(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Validate and convert inputs"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        if y is not None:
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples, got {X.shape[0]} and {y.shape[0]}"
                )

            # Check class labels
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                raise ValueError("y must contain at least 2 classes")

            return X, y

        return X

    def _compute_focal_weights(self, y: np.ndarray, alpha: float = 1.0, gamma: float = 2.0) -> np.ndarray:
        """
        Compute focal loss sample weights.

        Args:
            y: Class labels
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss

        Returns:
            Sample weights
        """
        class_counts = np.bincount(y)
        total_samples = len(y)

        # Compute class frequencies
        class_freq = class_counts / total_samples

        # Compute inverse class frequency weights
        inverse_freq = 1.0 / (class_freq + 1e-10)

        # Normalize to sum to number of samples
        normalized_weights = inverse_freq / np.sum(inverse_freq) * total_samples

        # Compute focal weights
        sample_weights = normalized_weights[y]

        # Apply gamma modulation
        sample_weights = sample_weights * (1 - class_freq[y]) ** gamma

        return sample_weights

    def _optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            X: Training features
            y: Training labels
            sample_weights: Sample weights

        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'max_iter': trial.suggest_int('max_iter', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_categorical('max_depth', [None, 3, 5, 7, 9]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
            }

            # Create model with trial parameters
            model = HistGradientBoostingClassifier(
                **params,
                early_stopping=self.config.early_stopping,
                validation_fraction=self.training_config.validation_fraction,
                n_iter_no_change=self.config.n_iter_no_change,
                class_weight=self.config.class_weight,
                random_state=self.training_config.random_state
            )

            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(
                    n_splits=self.training_config.cv_folds,
                    shuffle=True,
                    random_state=self.training_config.random_state
                ),
                scoring=self.training_config.cv_scoring,
                n_jobs=self.training_config.n_jobs
            )

            return cv_scores.mean()

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.training_config.random_state)
        )

        # Optimize
        study.optimize(
            objective,
            n_trials=self.training_config.optuna_trials,
            timeout=self.training_config.optuna_timeout
        )

        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        if not self.fitted:
            return {
                'fitted': False,
                'model_type': self.config.model_type
            }

        info = {
            'fitted': True,
            'model_type': self.config.model_type,
            'n_features': self.model.n_features_in_,
            'n_classes': self.model.n_classes_,
            'class_counts': self.class_counts.tolist(),
            'n_iter': self.model.n_iter_,
            'train_score': self.model.train_score_[-1] if hasattr(self.model, 'train_score_') else None,
            'best_iter': getattr(self.model, 'best_iter_', None)
        }

        if self.feature_names is not None:
            info['feature_names'] = self.feature_names

        return info


def features_to_array(features_list: list) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Convert list of AllFeatures to numpy arrays for training.

    Args:
        features_list: List of AllFeatures objects with labels

    Returns:
        Tuple of (X, y, feature_names)
    """
    X_list = []
    y_list = []

    for features, label in features_list:
        X_list.append(features.to_array())
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    # Get feature names from first item
    if features_list:
        feature_names = features_list[0][0].get_feature_names()
    else:
        feature_names = []

    return X, y, feature_names


class ModelTrainer:
    """Trainer class for threat classification models"""

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or get_config().training

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        model_config: Optional[ModelConfig] = None
    ) -> ThreatClassificationModel:
        """
        Train a threat classification model.

        Args:
            X: Training features
            y: Training labels
            feature_names: Optional list of feature names
            model_config: Model configuration

        Returns:
            Trained ThreatClassificationModel
        """
        # Initialize model
        model = ThreatClassificationModel(model_config)

        # Train
        model.fit(X, y, feature_names)

        # Save if configured
        if self.config.save_model and self.config.save_path:
            model.save_model(self.config.save_path)

        return model

    def evaluate_model(
        self,
        model: ThreatClassificationModel,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate trained model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        # AUC if binary classification
        if len(np.unique(y_test)) == 2:
            metrics['auc_roc'] = roc_auc_score(y_test, y_proba[:, 1])

        return metrics