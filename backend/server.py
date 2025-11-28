from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Cookie, Response
from fastapi.responses import StreamingResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import json
import base64
from collections import defaultdict, deque
import requests
import shutil

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Load YOLO model
MODEL_PATH = ROOT_DIR / "best.pt"
model = YOLO(str(MODEL_PATH))

# Create directories for uploads and processed videos
UPLOADS_DIR = ROOT_DIR / "uploads"
PROCESSED_DIR = ROOT_DIR / "processed"
UPLOADS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

# Extract clean class names from model (handle metadata in names)
MODEL_CLASS_NAMES = model.names
CLASS_NAMES = {}
for idx, name in MODEL_CLASS_NAMES.items():
    if idx < 7:  # Only first 7 are actual vehicle classes
        clean_name = name.split('-')[2].strip() if '-' in name else name
        CLASS_NAMES[idx] = clean_name
    else:
        break

logger.info(f"Loaded YOLOv8 model with classes: {CLASS_NAMES}")

# Tracking data
tracker_data = defaultdict(lambda: {'positions': deque(maxlen=30), 'speeds': deque(maxlen=10), 'class': None})

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Define Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Violation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    video_id: str
    violation_type: str  # 'no_helmet', 'wrong_way', 'overspeeding'
    timestamp: float  # Frame timestamp in seconds
    track_id: int
    speed: Optional[float] = None
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Video(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    filename: str
    original_path: str
    processed_path: Optional[str] = None
    status: str  # 'uploading', 'processing', 'completed', 'failed'
    total_violations: int = 0
    duration: float = 0.0
    fps: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CalibrationZone(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    reference_distance: float  # meters
    pixel_points: List[List[float]]  # [[x1, y1], [x2, y2]] for reference line
    speed_limit: float  # km/h
    direction_zone: Optional[List[List[float]]] = None  # Polygon points for direction
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Auth helper functions
async def get_current_user(session_token: Optional[str] = None) -> Optional[User]:
    if not session_token:
        return None
    
    session = await db.user_sessions.find_one({"session_token": session_token})
    if not session:
        return None
    
    # Check expiry
    expires_at = session['expires_at']
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    
    # Ensure expires_at has timezone info
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        await db.user_sessions.delete_one({"session_token": session_token})
        return None
    
    user = await db.users.find_one({"id": session['user_id']}, {"_id": 0})
    if user:
        if isinstance(user.get('created_at'), str):
            user['created_at'] = datetime.fromisoformat(user['created_at'])
        return User(**user)
    return None

# Auth endpoints
@api_router.post("/auth/session")
async def create_session(session_id: str, response: Response):
    """Process session_id from Emergent Auth and create user session"""
    try:
        # Call Emergent Auth API
        headers = {"X-Session-ID": session_id}
        auth_response = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers=headers
        )
        
        if auth_response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session ID")
        
        user_data = auth_response.json()
        
        # Check if user exists
        existing_user = await db.users.find_one({"email": user_data['email']}, {"_id": 0})
        if not existing_user:
            # Create new user
            new_user = User(
                email=user_data['email'],
                name=user_data['name'],
                picture=user_data.get('picture')
            )
            user_dict = new_user.model_dump()
            user_dict['created_at'] = user_dict['created_at'].isoformat()
            await db.users.insert_one(user_dict)
            user_id = new_user.id
        else:
            user_id = existing_user['id']
        
        # Create session
        session_token = user_data['session_token']
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        
        session_dict = session.model_dump()
        session_dict['expires_at'] = session_dict['expires_at'].isoformat()
        session_dict['created_at'] = session_dict['created_at'].isoformat()
        await db.user_sessions.insert_one(session_dict)
        
        # Set cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=True,
            samesite="none",
            max_age=7 * 24 * 60 * 60,
            path="/"
        )
        
        return {"success": True, "user_id": user_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/auth/me")
async def get_me(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@api_router.post("/auth/logout")
async def logout(session_token: Optional[str] = Cookie(None), response: Response = None):
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    response.delete_cookie(key="session_token", path="/")
    return {"success": True}

# Video upload and processing
@api_router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...), session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Save uploaded file
    video_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    file_path = UPLOADS_DIR / f"{video_id}{file_extension}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get video properties
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    # Create video record
    video = Video(
        id=video_id,
        user_id=user.id,
        filename=file.filename,
        original_path=str(file_path),
        status="uploaded",
        duration=duration,
        fps=fps
    )
    
    video_dict = video.model_dump()
    video_dict['created_at'] = video_dict['created_at'].isoformat()
    await db.videos.insert_one(video_dict)
    
    return video

@api_router.post("/videos/{video_id}/process")
async def process_video(video_id: str, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get video
    video = await db.videos.find_one({"id": video_id, "user_id": user.id}, {"_id": 0})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Start processing in background
    asyncio.create_task(process_video_background(video_id, user.id))
    
    return {"status": "processing", "video_id": video_id}

async def process_video_background(video_id: str, user_id: str):
    """Process video with YOLOv8 and detect violations"""
    try:
        # Update status
        await db.videos.update_one(
            {"id": video_id},
            {"$set": {"status": "processing"}}
        )
        
        video = await db.videos.find_one({"id": video_id}, {"_id": 0})
        input_path = video['original_path']
        output_path = str(PROCESSED_DIR / f"{video_id}_processed.mp4")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        violations_count = 0
        tracker_data = defaultdict(lambda: {'positions': deque(maxlen=30), 'class': None, 'violations': set()})
        
        # Get calibration (default if none exists)
        calibration = await db.calibration_zones.find_one({"user_id": user_id}, {"_id": 0})
        pixels_per_meter = 50  # Default
        speed_limit = 60  # km/h
        
        if calibration and calibration.get('reference_distance'):
            ref_points = calibration['pixel_points']
            pixel_distance = np.sqrt((ref_points[1][0] - ref_points[0][0])**2 + 
                                    (ref_points[1][1] - ref_points[0][1])**2)
            pixels_per_meter = pixel_distance / calibration['reference_distance']
            speed_limit = calibration.get('speed_limit', 60)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 detection with tracking
            results = model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    x1, y1, x2, y2 = box
                    class_name = CLASS_NAMES.get(int(cls), 'unknown')
                    
                    # Calculate center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Update tracker
                    tracker_data[track_id]['positions'].append((center_x, center_y, frame_idx))
                    tracker_data[track_id]['class'] = class_name
                    
                    # Calculate speed if enough positions
                    speed = None
                    if len(tracker_data[track_id]['positions']) >= 5:
                        positions = list(tracker_data[track_id]['positions'])
                        first_pos = positions[0]
                        last_pos = positions[-1]
                        
                        distance_pixels = np.sqrt((last_pos[0] - first_pos[0])**2 + 
                                                (last_pos[1] - first_pos[1])**2)
                        distance_meters = distance_pixels / pixels_per_meter
                        time_diff = (last_pos[2] - first_pos[2]) / fps
                        
                        if time_diff > 0:
                            speed = (distance_meters / time_diff) * 3.6  # m/s to km/h
                    
                    # Detect violations
                    violation_type = None
                    
                    # No helmet detection (rider without person nearby)
                    if class_name == 'rider':
                        # Check if person detected near rider
                        has_helmet = False
                        for other_id, other_data in tracker_data.items():
                            if other_data['class'] == 'person' and len(other_data['positions']) > 0:
                                other_pos = other_data['positions'][-1]
                                dist = np.sqrt((center_x - other_pos[0])**2 + (center_y - other_pos[1])**2)
                                if dist < 100:  # pixels
                                    has_helmet = True
                                    break
                        
                        if not has_helmet and 'no_helmet' not in tracker_data[track_id]['violations']:
                            violation_type = 'no_helmet'
                            tracker_data[track_id]['violations'].add('no_helmet')
                    
                    # Overspeeding detection
                    if speed and speed > speed_limit and 'overspeeding' not in tracker_data[track_id]['violations']:
                        violation_type = 'overspeeding'
                        tracker_data[track_id]['violations'].add('overspeeding')
                    
                    # Save violation to database
                    if violation_type:
                        violation = Violation(
                            user_id=user_id,
                            video_id=video_id,
                            violation_type=violation_type,
                            timestamp=frame_idx / fps,
                            track_id=int(track_id),
                            speed=speed,
                            confidence=float(conf),
                            bbox=[float(x1), float(y1), float(x2), float(y2)]
                        )
                        
                        violation_dict = violation.model_dump()
                        violation_dict['created_at'] = violation_dict['created_at'].isoformat()
                        await db.violations.insert_one(violation_dict)
                        violations_count += 1
                    
                    # Draw on frame
                    color = (0, 0, 255) if violation_type else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    label = f"{class_name} #{track_id}"
                    if speed:
                        label += f" {speed:.1f}km/h"
                    if violation_type:
                        label += f" [{violation_type.upper()}]"
                    
                    cv2.putText(frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        
        # Update video status
        await db.videos.update_one(
            {"id": video_id},
            {"$set": {
                "status": "completed",
                "processed_path": output_path,
                "total_violations": violations_count
            }}
        )
        
    except Exception as e:
        await db.videos.update_one(
            {"id": video_id},
            {"$set": {"status": "failed"}}
        )
        logger.error(f"Error processing video: {str(e)}")

@api_router.get("/videos")
async def get_videos(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    videos = await db.videos.find({"user_id": user.id}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return videos

@api_router.get("/videos/{video_id}")
async def get_video(video_id: str, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    video = await db.videos.find_one({"id": video_id, "user_id": user.id}, {"_id": 0})
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@api_router.get("/videos/{video_id}/download")
async def download_processed_video(video_id: str, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    video = await db.videos.find_one({"id": video_id, "user_id": user.id}, {"_id": 0})
    if not video or not video.get('processed_path'):
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    return FileResponse(video['processed_path'], media_type="video/mp4", filename=f"processed_{video['filename']}")

@api_router.get("/violations")
async def get_violations(video_id: Optional[str] = None, session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    query = {"user_id": user.id}
    if video_id:
        query["video_id"] = video_id
    
    violations = await db.violations.find(query, {"_id": 0}).sort("created_at", -1).to_list(1000)
    return violations

class CalibrationRequest(BaseModel):
    name: str
    reference_distance: float
    pixel_points: List[List[float]]
    speed_limit: float

@api_router.post("/calibration")
async def create_calibration(
    request: CalibrationRequest,
    session_token: Optional[str] = Cookie(None)
):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    calibration = CalibrationZone(
        user_id=user.id,
        name=request.name,
        reference_distance=request.reference_distance,
        pixel_points=request.pixel_points,
        speed_limit=request.speed_limit
    )
    
    calib_dict = calibration.model_dump()
    calib_dict['created_at'] = calib_dict['created_at'].isoformat()
    await db.calibration_zones.insert_one(calib_dict)
    
    return calibration

@api_router.get("/calibration")
async def get_calibration(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    calibration = await db.calibration_zones.find_one({"user_id": user.id}, {"_id": 0})
    return calibration if calibration else None

@api_router.get("/stats")
async def get_stats(session_token: Optional[str] = Cookie(None)):
    user = await get_current_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    total_videos = await db.videos.count_documents({"user_id": user.id})
    total_violations = await db.violations.count_documents({"user_id": user.id})
    
    # Count by type
    pipeline = [
        {"$match": {"user_id": user.id}},
        {"$group": {"_id": "$violation_type", "count": {"$sum": 1}}}
    ]
    violations_by_type = await db.violations.aggregate(pipeline).to_list(10)
    
    return {
        "total_videos": total_videos,
        "total_violations": total_violations,
        "violations_by_type": {item['_id']: item['count'] for item in violations_by_type}
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging already configured above

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
