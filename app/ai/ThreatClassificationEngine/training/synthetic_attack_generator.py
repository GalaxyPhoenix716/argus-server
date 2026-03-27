"""
Synthetic Attack Generator

Generates synthetic adversarial attacks for training threat classification models.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttackType(str, Enum):
    """Types of synthetic attacks"""
    GPS_SPOOFING = "gps_spoofing"
    SENSOR_INJECTION = "sensor_injection"
    TEMPORAL_MANIPULATION = "temporal_manipulation"
    MULTI_SENSOR_CORRUPTION = "multi_sensor_corruption"
    JAMMING = "jamming"
    POSITION_OFFSET = "position_offset"
    VELOCITY_MANIPULATION = "velocity_manipulation"
    SIGNAL_NOISE = "signal_noise"


class SyntheticAttackGenerator:
    """
    Generates synthetic adversarial attacks for training.

    Creates realistic attack patterns to augment training data for
    the threat classification model.
    """

    def __init__(
        self,
        random_state: int = 42,
        attack_intensity: float = 0.5
    ):
        """
        Initialize attack generator.

        Args:
            random_state: Random seed for reproducibility
            attack_intensity: Intensity of generated attacks [0, 1]
        """
        self.random_state = random_state
        self.attack_intensity = attack_intensity
        np.random.seed(random_state)

    def generate_attack(
        self,
        normal_data: np.ndarray,
        attack_type: AttackType,
        attack_duration: Optional[int] = None,
        start_position: Optional[int] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate synthetic attack on normal data.

        Args:
            normal_data: Normal telemetry data (timesteps, n_features)
            attack_type: Type of attack to generate
            attack_duration: Duration of attack in timesteps
            start_position: Start position of attack
            **kwargs: Attack-specific parameters

        Returns:
            Tuple of (attacked_data, attack_mask, attack_metadata)
        """
        if normal_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {normal_data.shape}")

        n_timesteps, n_features = normal_data.shape

        # Set default attack duration
        if attack_duration is None:
            attack_duration = max(10, n_timesteps // 10)

        # Set default start position
        if start_position is None:
            start_position = np.random.randint(0, n_timesteps - attack_duration)

        # Generate attack mask (1 = attacked, 0 = normal)
        attack_mask = np.zeros(n_timesteps, dtype=int)
        attack_mask[start_position:start_position + attack_duration] = 1

        # Generate attacked data
        attacked_data = normal_data.copy()

        # Apply attack based on type
        if attack_type == AttackType.GPS_SPOOFING:
            attacked_data = self._apply_gps_spoofing(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.SENSOR_INJECTION:
            attacked_data = self._apply_sensor_injection(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.TEMPORAL_MANIPULATION:
            attacked_data = self._apply_temporal_manipulation(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.MULTI_SENSOR_CORRUPTION:
            attacked_data = self._apply_multi_sensor_corruption(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.JAMMING:
            attacked_data = self._apply_jamming(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.POSITION_OFFSET:
            attacked_data = self._apply_position_offset(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.VELOCITY_MANIPULATION:
            attacked_data = self._apply_velocity_manipulation(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        elif attack_type == AttackType.SIGNAL_NOISE:
            attacked_data = self._apply_signal_noise(
                attacked_data, attack_mask, start_position, attack_duration, **kwargs
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Create metadata
        metadata = {
            'attack_type': attack_type.value,
            'start_position': start_position,
            'duration': attack_duration,
            'intensity': self.attack_intensity,
            'n_features': n_features,
            'n_timesteps': n_timesteps
        }

        return attacked_data, attack_mask, metadata

    def _apply_gps_spoofing(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        position_offset: float = None,
        velocity_mismatch: float = None,
        **kwargs
    ) -> np.ndarray:
        """Apply GPS spoofing attack"""
        attacked_data = data.copy()

        # Default parameters
        if position_offset is None:
            position_offset = 0.1 * self.attack_intensity
        if velocity_mismatch is None:
            velocity_mismatch = 0.2 * self.attack_intensity

        # Assume position features are in columns 0, 1
        # Velocity features are in columns 2, 3
        if data.shape[1] >= 4:
            # Apply position offset
            attacked_data[start:start + duration, 0] += position_offset
            attacked_data[start:start + duration, 1] += position_offset

            # Apply velocity mismatch
            attacked_data[start:start + duration, 2] += velocity_mismatch
            attacked_data[start:start + duration, 3] += velocity_mismatch

        return attacked_data

    def _apply_sensor_injection(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        injected_pattern: str = 'sine',
        frequency: float = None,
        amplitude: float = None,
        **kwargs
    ) -> np.ndarray:
        """Apply sensor injection attack"""
        attacked_data = data.copy()

        # Default parameters
        if frequency is None:
            frequency = 0.1 * self.attack_intensity
        if amplitude is None:
            amplitude = 0.05 * self.attack_intensity

        # Apply synthetic pattern to first sensor
        t = np.arange(duration)
        if injected_pattern == 'sine':
            pattern = amplitude * np.sin(2 * np.pi * frequency * t)
        elif injected_pattern == 'random':
            pattern = amplitude * np.random.randn(duration)
        else:
            pattern = amplitude * np.ones(duration)

        attacked_data[start:start + duration, 0] = pattern

        return attacked_data

    def _apply_temporal_manipulation(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        time_shift: int = None,
        data_reorder: bool = False,
        **kwargs
    ) -> np.ndarray:
        """Apply temporal manipulation attack"""
        attacked_data = data.copy()

        # Default parameters
        if time_shift is None:
            time_shift = int(duration * 0.2)

        # Time shift attack
        if time_shift != 0:
            # Shift data within the attack window
            attacked_data[start:start + duration] = np.roll(
                attacked_data[start:start + duration],
                time_shift,
                axis=0
            )

        # Data reordering
        if data_reorder:
            # Reverse the attack window
            attacked_data[start:start + duration] = attacked_data[start:start + duration][::-1]

        return attacked_data

    def _apply_multi_sensor_corruption(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        corruption_level: float = None,
        correlated_noise: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Apply multi-sensor corruption attack"""
        attacked_data = data.copy()

        # Default parameters
        if corruption_level is None:
            corruption_level = 0.1 * self.attack_intensity

        # Correlated noise
        if correlated_noise:
            noise = corruption_level * np.random.randn(duration)
            # Apply same noise to multiple sensors
            for col in range(min(5, data.shape[1])):
                attacked_data[start:start + duration, col] += noise
        else:
            # Independent noise per sensor
            attacked_data[start:start + duration] += (
                corruption_level * np.random.randn(duration, data.shape[1])
            )

        return attacked_data

    def _apply_jamming(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        noise_level: float = None,
        persistent: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Apply jamming attack"""
        attacked_data = data.copy()

        # Default parameters
        if noise_level is None:
            noise_level = 0.3 * self.attack_intensity

        # Add high-frequency noise
        t = np.arange(duration)
        high_freq_noise = noise_level * np.random.randn(duration)

        # Apply to all features
        attacked_data[start:start + duration] += high_freq_noise[:, np.newaxis]

        return attacked_data

    def _apply_position_offset(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        offset_magnitude: float = None,
        **kwargs
    ) -> np.ndarray:
        """Apply position offset attack"""
        attacked_data = data.copy()

        # Default parameters
        if offset_magnitude is None:
            offset_magnitude = 0.05 * self.attack_intensity

        # Apply offset to position features (assuming columns 0, 1)
        if data.shape[1] >= 2:
            attacked_data[start:start + duration, 0] += offset_magnitude
            attacked_data[start:start + duration, 1] += offset_magnitude

        return attacked_data

    def _apply_velocity_manipulation(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        velocity_offset: float = None,
        **kwargs
    ) -> np.ndarray:
        """Apply velocity manipulation attack"""
        attacked_data = data.copy()

        # Default parameters
        if velocity_offset is None:
            velocity_offset = 0.1 * self.attack_intensity

        # Apply offset to velocity features (assuming columns 2, 3)
        if data.shape[1] >= 4:
            attacked_data[start:start + duration, 2] += velocity_offset
            attacked_data[start:start + duration, 3] += velocity_offset

        return attacked_data

    def _apply_signal_noise(
        self,
        data: np.ndarray,
        attack_mask: np.ndarray,
        start: int,
        duration: int,
        noise_std: float = None,
        **kwargs
    ) -> np.ndarray:
        """Apply signal noise attack"""
        attacked_data = data.copy()

        # Default parameters
        if noise_std is None:
            noise_std = 0.2 * self.attack_intensity

        # Add Gaussian noise
        attacked_data[start:start + duration] += (
            noise_std * np.random.randn(duration, data.shape[1])
        )

        return attacked_data

    def generate_dataset(
        self,
        normal_data: np.ndarray,
        attack_types: List[AttackType],
        n_attacks: int,
        attack_duration_range: Tuple[int, int] = (10, 50)
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Generate synthetic dataset with attacks.

        Args:
            normal_data: Normal telemetry data (timesteps, n_features)
            attack_types: List of attack types to generate
            n_attacks: Number of attacks to generate
            attack_duration_range: (min, max) duration range

        Returns:
            Tuple of (augmented_data, attack_labels, attack_masks, metadata_list)
        """
        if len(attack_types) == 0:
            raise ValueError("attack_types cannot be empty")

        augmented_data = []
        attack_labels = []
        attack_masks = []
        metadata_list = []

        n_timesteps, n_features = normal_data.shape

        logger.info(f"Generating {n_attacks} attacks from {len(attack_types)} types")

        for i in range(n_attacks):
            # Randomly select attack type
            attack_type = np.random.choice(attack_types)

            # Random duration
            min_dur, max_dur = attack_duration_range
            attack_duration = np.random.randint(min_dur, min(max_dur + 1, n_timesteps))

            # Random start position
            start_position = np.random.randint(0, n_timesteps - attack_duration)

            # Generate attack
            attacked_data, attack_mask, metadata = self.generate_attack(
                normal_data=normal_data,
                attack_type=attack_type,
                attack_duration=attack_duration,
                start_position=start_position
            )

            # Label (1 = attack, 0 = normal)
            label = 1

            # Store
            augmented_data.append(attacked_data)
            attack_labels.append(label)
            attack_masks.append(attack_mask)
            metadata_list.append(metadata)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{n_attacks} attacks")

        # Combine all data
        all_data = np.concatenate(augmented_data, axis=0)
        all_labels = np.concatenate(attack_labels, axis=0)
        all_masks = np.concatenate(attack_masks, axis=0)

        logger.info(f"Generated dataset: {all_data.shape[0]} timesteps")

        return all_data, all_labels, all_masks, metadata_list

    def generate_failure_data(
        self,
        normal_data: np.ndarray,
        failure_types: List[str],
        n_failures: int,
        failure_severity: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Generate synthetic failure data.

        Args:
            normal_data: Normal telemetry data
            failure_types: List of failure types
            n_failures: Number of failures to generate
            failure_severity: Severity of failures [0, 1]

        Returns:
            Tuple of (failed_data, failure_labels, failure_masks, metadata)
        """
        # Similar to attack generation but for failures
        # For brevity, using simplified implementation

        failed_data = []
        failure_labels = []
        failure_masks = []
        metadata_list = []

        n_timesteps, n_features = normal_data.shape

        for i in range(n_failures):
            failure_type = np.random.choice(failure_types)

            # Simulate failure
            failed_telemetry = normal_data.copy()

            if failure_type == 'sensor_drift':
                # Gradual degradation
                drift_rate = failure_severity * 0.01
                failed_telemetry += drift_rate * np.arange(n_timesteps)[:, np.newaxis]

            elif failure_type == 'sensor_stuck':
                # Stuck at last value
                stuck_value = normal_data[-1]
                failed_telemetry = np.tile(stuck_value, (n_timesteps, 1))

            elif failure_type == 'sensor_dead':
                # Zero values
                failed_telemetry = np.zeros_like(normal_data)

            # Failure mask (all timesteps)
            failure_mask = np.ones(n_timesteps, dtype=int)
            label = 0  # Label failures differently from attacks

            failed_data.append(failed_telemetry)
            failure_labels.append(label)
            failure_masks.append(failure_mask)
            metadata_list.append({
                'failure_type': failure_type,
                'severity': failure_severity,
                'n_timesteps': n_timesteps
            })

        # Combine
        all_data = np.concatenate(failed_data, axis=0)
        all_labels = np.concatenate(failure_labels, axis=0)
        all_masks = np.concatenate(failure_masks, axis=0)

        return all_data, all_labels, all_masks, metadata_list