"""
Classifier Trainer

Trainer for threat classification models with hyperparameter tuning,
cross-validation, and model persistence.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from pathlib import Path
import pickle
from datetime import datetime

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
import optuna

from app.ai.threat_model import ThreatClassificationModel, features_to_array
from app.ai.training.synthetic_attack_generator import SyntheticAttackGenerator, AttackType
from app.config.threat_config import TrainingConfig, ModelConfig, get_config

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """
    Trainer for threat classification models.

    Handles:
    - Data preprocessing and feature extraction
    - Hyperparameter tuning with Optuna
    - Cross-validation
    - Model persistence
    - Evaluation metrics
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model_config: Model configuration
        """
        self.config = config or get_config().training
        self.model_config = model_config or get_config().model

        # Initialize components
        self.attack_generator = SyntheticAttackGenerator(
            random_state=self.config.random_state
        )

        # State
        self.model = None
        self.feature_names = None
        self.training_history = {}

    def prepare_training_data(
        self,
        normal_data: np.ndarray,
        n_synthetic_attacks: int = 1000,
        n_synthetic_failures: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data by augmenting with synthetic attacks and failures.

        Args:
            normal_data: Normal telemetry data (timesteps, n_features)
            n_synthetic_attacks: Number of synthetic attacks to generate
            n_synthetic_failures: Number of synthetic failures to generate

        Returns:
            Tuple of (features, labels, feature_names)
        """
        logger.info("Preparing training data with synthetic attacks and failures")

        # Generate synthetic attacks
        attack_types = [
            AttackType.GPS_SPOOFING,
            AttackType.SENSOR_INJECTION,
            AttackType.TEMPORAL_MANIPULATION,
            AttackType.MULTI_SENSOR_CORRUPTION,
            AttackType.JAMMING,
            AttackType.POSITION_OFFSET,
            AttackType.VELOCITY_MANIPULATION,
            AttackType.SIGNAL_NOISE
        ]

        attacked_data, attack_labels, attack_masks, attack_metadata = (
            self.attack_generator.generate_dataset(
                normal_data=normal_data,
                attack_types=attack_types,
                n_attacks=n_synthetic_attacks
            )
        )

        # Generate synthetic failures
        failure_types = ['sensor_drift', 'sensor_stuck', 'sensor_dead']
        failed_data, failure_labels, failure_masks, failure_metadata = (
            self.attack_generator.generate_failure_data(
                normal_data=normal_data,
                failure_types=failure_types,
                n_failures=n_synthetic_failures
            )
        )

        # Combine normal data with labels
        normal_labels = np.zeros(normal_data.shape[0])

        # Combine all data
        all_data = np.concatenate([
            normal_data,
            attacked_data,
            failed_data
        ], axis=0)

        all_labels = np.concatenate([
            normal_labels,
            attack_labels,
            failure_labels
        ])

        # Convert labels to classification format
        # 0: normal, 1: failure, 2: attack (simplified for now)
        classification_labels = np.zeros(len(all_labels))
        classification_labels[len(normal_data):len(normal_data) + len(attack_labels)] = 2  # Attack
        classification_labels[len(normal_data) + len(attack_labels):] = 1  # Failure

        logger.info(
            f"Training data prepared: {len(all_data)} samples "
            f"({np.sum(classification_labels == 0)} normal, "
            f"{np.sum(classification_labels == 1)} failures, "
            f"{np.sum(classification_labels == 2)} attacks)"
        )

        return all_data, classification_labels, [f"feature_{i}" for i in range(all_data.shape[1])]

    def train_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_config: Optional[ModelConfig] = None
    ) -> ThreatClassificationModel:
        """
        Train threat classification model.

        Args:
            X: Training features
            y: Training labels
            feature_names: List of feature names
            model_config: Model configuration

        Returns:
            Trained ThreatClassificationModel
        """
        logger.info(f"Training model on {X.shape[0]} samples")

        if model_config is None:
            model_config = self.model_config

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Initialize model
        self.model = ThreatClassificationModel(
            model_config=model_config,
            training_config=self.config
        )

        # Train model
        self.model.fit(X, y, self.feature_names)

        # Store training history
        self.training_history = {
            'training_samples': X.shape[0],
            'n_features': X.shape[1],
            'class_distribution': {
                int(cls): int(np.sum(y == cls))
                for cls in np.unique(y)
            },
            'model_info': self.model.get_model_info(),
            'trained_at': datetime.now().isoformat()
        }

        logger.info("Model training completed")

        return self.model

    def hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using Optuna.

        Args:
            X: Training features
            y: Training labels
            n_trials: Number of optimization trials

        Returns:
            Best hyperparameters
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

        def objective(trial):
            # Define search space
            params = {
                'max_iter': trial.suggest_int('max_iter', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_categorical('max_depth', [None, 3, 5, 7, 9]),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 50),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 10.0),
            }

            # Create model
            from sklearn.ensemble import HistGradientBoostingClassifier
            model = HistGradientBoostingClassifier(
                **params,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                class_weight='balanced',
                random_state=self.config.random_state
            )

            # Cross-validation
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state
            )

            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=self.config.cv_scoring,
                n_jobs=self.config.n_jobs
            )

            return scores.mean()

        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state)
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=self.config.optuna_timeout
        )

        logger.info(
            f"Best score: {study.best_value:.4f} "
            f"with params: {study.best_params}"
        )

        return study.best_params

    def evaluate_model(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate trained model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None or not self.model.fitted:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating model on test set")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        # Metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        )

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        # AUC (for each class)
        n_classes = len(np.unique(y_test))
        for i in range(n_classes):
            if len(np.unique(y_test)) == 2 and n_classes == 2:
                # Binary case
                metrics['auc_roc'] = roc_auc_score(y_test, y_proba[:, 1])
                break
            else:
                # Multi-class case - use one-vs-rest
                from sklearn.metrics import roc_auc_score
                y_test_binary = (y_test == i).astype(int)
                if len(np.unique(y_test_binary)) > 1:
                    metrics[f'auc_roc_class_{i}'] = roc_auc_score(y_test_binary, y_proba[:, i])

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report

        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def save_model(self, filepath: str):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(filepath))

        # Save metadata
        metadata_path = filepath.with_suffix('.metadata.pkl')
        metadata = {
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'config': self.config,
            'model_config': self.model_config
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> Tuple['ClassifierTrainer', ThreatClassificationModel]:
        """
        Load trained model and trainer.

        Args:
            filepath: Path to saved model

        Returns:
            Tuple of (trainer, model)
        """
        # Load model
        model = ThreatClassificationModel.load_model(filepath)

        # Load metadata
        metadata_path = Path(filepath).with_suffix('.metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create trainer
        trainer = cls(
            config=metadata['config'],
            model_config=metadata['model_config']
        )
        trainer.model = model
        trainer.feature_names = metadata['feature_names']
        trainer.training_history = metadata['training_history']

        logger.info(f"Model loaded from {filepath}")

        return trainer, model

    def get_feature_importance(self, top_k: Optional[int] = None) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model must be trained first")

        return self.model.get_feature_importance(top_k=top_k)

    def get_training_info(self) -> Dict[str, Any]:
        """Get training information"""
        if not self.training_history:
            return {}

        return {
            'training_samples': self.training_history.get('training_samples'),
            'n_features': self.training_history.get('n_features'),
            'class_distribution': self.training_history.get('class_distribution'),
            'trained_at': self.training_history.get('trained_at'),
            'model_info': self.training_history.get('model_info')
        }

    def incremental_training(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray
    ) -> ThreatClassificationModel:
        """
        Perform incremental training on new data.

        Args:
            X_new: New training features
            y_new: New training labels

        Returns:
            Updated model
        """
        if self.model is None:
            # Initial training
            return self.train_model(X_new, y_new)

        # Note: HistGradientBoostingClassifier doesn't support
        # partial_fit, so this would require re-training with
        # all data in production
        logger.warning("Incremental training requires re-training with all data")

        return self.model

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Labels
            cv_folds: Number of CV folds

        Returns:
            Cross-validation scores
        """
        if self.model is None:
            # Create untrained model for CV
            self.model = ThreatClassificationModel(
                model_config=self.model_config,
                training_config=self.config
            )

        cv_folds = cv_folds or self.config.cv_folds

        from sklearn.model_selection import cross_validate

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        scoring = [
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro'
        ]

        scores = cross_validate(
            self.model.model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=self.config.n_jobs
        )

        cv_results = {
            f'test_{metric}': float(np.mean(scores[f'test_{metric}']))
            for metric in scoring
        }

        logger.info(f"CV Accuracy: {cv_results['test_accuracy']:.4f} (+/- {np.std(scores['test_accuracy']):.4f})")

        return cv_results