import logging
from fastapi import APIRouter, HTTPException
from firebase_admin import firestore

from app.models.mission import Mission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/mission", tags=["mission"])

@router.get("/", response_model=list[Mission])
async def list_missions():
    """Fetch all available missions from Firestore."""
    try:
        db = firestore.client()
        docs = db.collection("missions").stream()
        
        missions = []
        for doc in docs:
            missions.append(Mission.from_firestore(doc.id, doc.to_dict()))
            
        return missions
    except Exception as e:
        logger.error(f"Error fetching missions: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred")
