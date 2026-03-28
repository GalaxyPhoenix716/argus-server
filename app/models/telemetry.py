"""
Telemetry domain model.

Represents a single 25-channel TelemetryFrame as received in real-time from the
Argus Telemetry Simulator via WebSocket.  This object is NEVER persisted to
Firestore — it is forwarded straight to the frontend and optionally used by the
AI anomaly-detection layer.

Channel order (matches simulator's .to_array() output):
  0  position_lat        degrees     −90  →  +90
  1  position_lon        degrees     −180 →  +180
  2  velocity_x          km/s        ±8
  3  velocity_y          km/s        ±8
  4  velocity_z          km/s        ±8
  5  altitude            m           ~600 000 (LEO)
  6  acceleration_x      m/s²        ±10
  7  acceleration_y      m/s²        ±10
  8  acceleration_z      m/s²        ±10
  9  temperature         °C          −50 → +80
  10 pressure            Pa          1 → 101325
  11 humidity            %           0 or 50
  12 battery_level       %           0 → 100
  13 signal_strength     dBm         −120 → −20
  14 gyro_x              rad/s       ~0 ± small
  15 gyro_y              rad/s       ~0 ± small
  16 gyro_z              rad/s       ~0 ± small
  17 magnetometer_x      µT          −50 → +50
  18 magnetometer_y      µT          −50 → +50
  19 magnetometer_z      µT          −50 → +50
  20 attitude_roll       degrees     −180 → +180
  21 attitude_pitch      degrees     −90  → +90
  22 attitude_yaw        degrees     −180 → +180
  23 angular_velocity    rad/s       0 → ~5
  24 timestamp           s (Unix)    monotonically increasing
"""

from pydantic import BaseModel, Field


class TelemetryFrame(BaseModel):
    """
    A single real-time telemetry snapshot from the satellite simulator.
    Immutable once created — the server only ever forwards these downstream.
    """

    # ── Position & Velocity ───────────────────────────────────────────────────
    position_lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    position_lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")
    velocity_x: float = Field(..., description="Velocity X component in km/s")
    velocity_y: float = Field(..., description="Velocity Y component in km/s")
    velocity_z: float = Field(..., description="Velocity Z component in km/s")
    altitude: float = Field(..., ge=0.0, description="Altitude in metres")

    # ── Acceleration ──────────────────────────────────────────────────────────
    acceleration_x: float = Field(..., description="Acceleration X in m/s²")
    acceleration_y: float = Field(..., description="Acceleration Y in m/s²")
    acceleration_z: float = Field(..., description="Acceleration Z in m/s²")

    # ── Environmental ─────────────────────────────────────────────────────────
    temperature: float = Field(..., description="Temperature in °C")
    pressure: float = Field(..., ge=1.0, description="Atmospheric pressure in Pa")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity in %")

    # ── Power & Communications ────────────────────────────────────────────────
    battery_level: float = Field(..., ge=0.0, le=100.0, description="Battery state-of-charge in %")
    signal_strength: float = Field(..., ge=-120.0, le=-20.0, description="Signal strength in dBm")

    # ── IMU (Gyroscope) ───────────────────────────────────────────────────────
    gyro_x: float = Field(..., description="Gyro X in rad/s")
    gyro_y: float = Field(..., description="Gyro Y in rad/s")
    gyro_z: float = Field(..., description="Gyro Z in rad/s")

    # ── Magnetometer ──────────────────────────────────────────────────────────
    magnetometer_x: float = Field(..., description="Magnetometer X in µT")
    magnetometer_y: float = Field(..., description="Magnetometer Y in µT")
    magnetometer_z: float = Field(..., description="Magnetometer Z in µT")

    # ── Attitude ──────────────────────────────────────────────────────────────
    attitude_roll: float = Field(..., ge=-180.0, le=180.0, description="Roll in degrees")
    attitude_pitch: float = Field(..., ge=-90.0, le=90.0, description="Pitch in degrees")
    attitude_yaw: float = Field(..., ge=-180.0, le=180.0, description="Yaw in degrees")
    angular_velocity: float = Field(..., ge=0.0, description="Angular velocity magnitude in rad/s")

    # ── Time ──────────────────────────────────────────────────────────────────
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds")

    # ── Helpers ───────────────────────────────────────────────────────────────

    CHANNELS: list[str] = [
        "position_lat", "position_lon",
        "velocity_x", "velocity_y", "velocity_z",
        "altitude",
        "acceleration_x", "acceleration_y", "acceleration_z",
        "temperature", "pressure", "humidity",
        "battery_level", "signal_strength",
        "gyro_x", "gyro_y", "gyro_z",
        "magnetometer_x", "magnetometer_y", "magnetometer_z",
        "attitude_roll", "attitude_pitch", "attitude_yaw",
        "angular_velocity",
        "timestamp",
    ]

    def to_array(self) -> list[float]:
        """Return channels as a 25-element float list (same order as simulator)."""
        return [getattr(self, ch) for ch in self.CHANNELS]

    @classmethod
    def from_dict(cls, data: dict) -> "TelemetryFrame":
        """Construct from a raw simulator dict."""
        return cls(**data)

    class Config:
        # Allows the model to be used as a value object in Mission
        frozen = True
