"""
Telemetry request/response Pydantic schemas for the API + WebSocket layer.

TelemetryFrameSchema  — the wire format for a single frame coming from the
                         simulator and going out to the frontend.  Mirrors
                         TelemetryFrame exactly so it can be used directly
                         on the WebSocket endpoint without an extra conversion
                         step.
"""

from pydantic import BaseModel, Field


class TelemetryFrameSchema(BaseModel):
    """
    Wire schema for a single 25-channel telemetry frame.
    Used for WebSocket messages (sim → server → frontend).
    """

    # ── Position & Velocity ───────────────────────────────────────────────────
    position_lat  : float = Field(..., ge=-90.0,   le=90.0,   description="Latitude (°)")
    position_lon  : float = Field(..., ge=-180.0,  le=180.0,  description="Longitude (°)")
    velocity_x    : float = Field(..., description="Velocity X (km/s)")
    velocity_y    : float = Field(..., description="Velocity Y (km/s)")
    velocity_z    : float = Field(..., description="Velocity Z (km/s)")
    altitude      : float = Field(..., ge=0.0,                description="Altitude (m)")

    # ── Acceleration ──────────────────────────────────────────────────────────
    acceleration_x: float = Field(..., description="Acceleration X (m/s²)")
    acceleration_y: float = Field(..., description="Acceleration Y (m/s²)")
    acceleration_z: float = Field(..., description="Acceleration Z (m/s²)")

    # ── Environmental ─────────────────────────────────────────────────────────
    temperature   : float = Field(..., description="Temperature (°C)")
    pressure      : float = Field(..., ge=1.0,                description="Pressure (Pa)")
    humidity      : float = Field(..., ge=0.0,   le=100.0,    description="Humidity (%)")

    # ── Power & Communications ────────────────────────────────────────────────
    battery_level   : float = Field(..., ge=0.0,   le=100.0,  description="Battery level (%)")
    signal_strength : float = Field(..., ge=-120.0, le=-20.0, description="Signal strength (dBm)")

    # ── IMU Gyroscope ─────────────────────────────────────────────────────────
    gyro_x: float = Field(..., description="Gyro X (rad/s)")
    gyro_y: float = Field(..., description="Gyro Y (rad/s)")
    gyro_z: float = Field(..., description="Gyro Z (rad/s)")

    # ── Magnetometer ──────────────────────────────────────────────────────────
    magnetometer_x: float = Field(..., description="Magnetometer X (µT)")
    magnetometer_y: float = Field(..., description="Magnetometer Y (µT)")
    magnetometer_z: float = Field(..., description="Magnetometer Z (µT)")

    # ── Attitude ──────────────────────────────────────────────────────────────
    attitude_roll    : float = Field(..., ge=-180.0, le=180.0, description="Roll (°)")
    attitude_pitch   : float = Field(..., ge=-90.0,  le=90.0,  description="Pitch (°)")
    attitude_yaw     : float = Field(..., ge=-180.0, le=180.0, description="Yaw (°)")
    angular_velocity : float = Field(..., ge=0.0,              description="Angular velocity magnitude (rad/s)")

    # ── Time ──────────────────────────────────────────────────────────────────
    timestamp: float = Field(..., description="Unix epoch timestamp (s)")

    class Config:
        # Accept both camelCase and snake_case from the simulator
        populate_by_name = True
