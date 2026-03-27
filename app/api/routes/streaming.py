"""
WebSocket Streaming Routes

Real-time streaming of anomaly, threat, and explanation events.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, List, Optional, Set
import logging
import json
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["streaming"])

# Connection manager
class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        # Store connections by channel_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Store all connections for broadcasting
        self.all_connections: Set[WebSocket] = set()

    async def connect(
        self,
        websocket: WebSocket,
        channel_id: Optional[str] = Query(default=None, description="Filter by channel ID")
    ):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.all_connections.add(websocket)

        if channel_id:
            if channel_id not in self.active_connections:
                self.active_connections[channel_id] = []
            self.active_connections[channel_id].append(websocket)
            logger.info(f"Client connected to channel {channel_id}")
        else:
            logger.info(f"Client connected (all channels)")

    def disconnect(self, websocket: WebSocket, channel_id: Optional[str] = None):
        """Remove WebSocket connection"""
        self.all_connections.discard(websocket)

        if channel_id and channel_id in self.active_connections:
            if websocket in self.active_connections[channel_id]:
                self.active_connections[channel_id].remove(websocket)
                if not self.active_connections[channel_id]:
                    del self.active_connections[channel_id]
            logger.info(f"Client disconnected from channel {channel_id}")
        else:
            logger.info("Client disconnected")

    async def send_personal_message(
        self,
        message: dict,
        websocket: WebSocket
    ):
        """Send message to specific connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast_to_channel(
        self,
        message: dict,
        channel_id: str
    ):
        """Broadcast message to all connections for a specific channel"""
        if channel_id not in self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections[channel_id]:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to channel {channel_id}: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn, channel_id)

    async def broadcast_to_all(
        self,
        message: dict
    ):
        """Broadcast message to all connections"""
        disconnected = []
        for connection in self.all_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.all_connections.discard(conn)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    channel_id: Optional[str] = Query(default=None)
):
    """
    WebSocket endpoint for real-time streaming.

    Clients can subscribe to:
    - All channels (no channel_id)
    - Specific channel (provide channel_id)

    Messages will be sent for:
    - Anomaly detection events
    - Threat classification events
    - Explanation generation events
    - Alert events
    """
    await manager.connect(websocket, channel_id)

    try:
        # Send welcome message
        await manager.send_personal_message(
            {
                "type": "connected",
                "timestamp": datetime.now().isoformat(),
                "channel_id": channel_id,
                "message": "Connected to ARGUS streaming service"
            },
            websocket
        )

        # Keep connection alive
        while True:
            # Wait for messages from client (ping/pong, subscriptions, etc.)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                # Handle ping
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        },
                        websocket
                    )

                # Handle subscription update
                elif message.get("type") == "subscribe":
                    new_channel = message.get("channel_id")
                    old_channel = channel_id
                    channel_id = new_channel

                    # Update subscription
                    await manager.send_personal_message(
                        {
                            "type": "subscribed",
                            "timestamp": datetime.now().isoformat(),
                            "channel_id": channel_id
                        },
                        websocket
                    )

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        manager.disconnect(websocket, channel_id)
        logger.info(f"WebSocket disconnected: {channel_id}")


# Utility functions for broadcasting events

async def broadcast_anomaly_event(
    channel_id: str,
    anomaly_data: dict
):
    """Broadcast anomaly detection event"""
    message = {
        "type": "anomaly",
        "timestamp": datetime.now().isoformat(),
        "channel_id": channel_id,
        "data": anomaly_data
    }
    await manager.broadcast_to_channel(message, channel_id)


async def broadcast_threat_event(
    channel_id: str,
    threat_data: dict
):
    """Broadcast threat classification event"""
    message = {
        "type": "threat",
        "timestamp": datetime.now().isoformat(),
        "channel_id": channel_id,
        "data": threat_data
    }
    await manager.broadcast_to_channel(message, channel_id)


async def broadcast_explanation_event(
    channel_id: str,
    explanation_data: dict
):
    """Broadcast explanation generation event"""
    message = {
        "type": "explanation",
        "timestamp": datetime.now().isoformat(),
        "channel_id": channel_id,
        "data": explanation_data
    }
    await manager.broadcast_to_channel(message, channel_id)


async def broadcast_alert_event(
    channel_id: str,
    alert_data: dict
):
    """Broadcast alert event"""
    message = {
        "type": "alert",
        "timestamp": datetime.now().isoformat(),
        "channel_id": channel_id,
        "data": alert_data
    }
    await manager.broadcast_to_channel(message, channel_id)


async def broadcast_system_event(
    event_type: str,
    event_data: dict,
    target_channel: Optional[str] = None
):
    """Broadcast system-wide event"""
    message = {
        "type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": event_data
    }

    if target_channel:
        await manager.broadcast_to_channel(message, target_channel)
    else:
        await manager.broadcast_to_all(message)


# Example usage functions (to be called by services)

def get_connection_count() -> Dict[str, int]:
    """Get number of active connections"""
    return {
        "total": len(manager.all_connections),
        "by_channel": {
            channel: len(connections)
            for channel, connections in manager.active_connections.items()
        }
    }


async def simulate_events():
    """
    Simulate streaming events (for testing/demo purposes).

    This function generates sample events to demonstrate the streaming capability.
    """
    import random

    channels = ["P-1", "P-2", "A-7", "A-9"]
    threat_classes = ["attack", "failure", "unknown"]
    patterns = ["spike", "drift", "persistent", "intermittent"]

    while True:
        # Randomly select a channel
        channel = random.choice(channels)

        # Generate anomaly event
        await broadcast_anomaly_event(channel, {
            "anomaly_score": random.uniform(0.5, 1.0),
            "window": [random.randint(100, 500), random.randint(600, 1000)]
        })

        await asyncio.sleep(2)

        # Generate threat event
        await broadcast_threat_event(channel, {
            "class": random.choice(threat_classes),
            "confidence": random.uniform(0.6, 0.95),
            "risk_score": random.uniform(0.1, 0.9),
            "specific_type": random.choice([
                "gps_spoofing", "sensor_drift", "communication_loss", None
            ])
        })

        await asyncio.sleep(2)

        # Generate explanation event
        await broadcast_explanation_event(channel, {
            "pattern": random.choice(patterns),
            "top_features": [
                {"name": f"feature_{i}", "importance": random.random()}
                for i in range(3)
            ]
        })

        await asyncio.sleep(5)