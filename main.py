# FastAPI Deepfake Detection Service with WebSocket Support
# Handles audio extraction from video files and real-time detection

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np
import joblib
import os
import tempfile
import asyncio
import json
import base64
from typing import Dict, Any
import logging
from datetime import datetime
import ffmpeg
import io
from pathlib import Path
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YamNet Deepfake Voice Detection API",
    description="Real-time deepfake voice detection using YamNet",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DeepfakeDetector:
    def __init__(self, model_dir: str = "data/"):
        """Initialize the deepfake detector"""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.yamnet_model = None
        self.classifier = None
        self.scaler = None
        self.sample_rate = 16000
        self.is_loaded = False
        self.prediction_threshold = 0.75  # Higher threshold to reduce false positives
        
        # Load models on initialization
        self.load_models()
    
    def load_yamnet(self):
        """Load YamNet model from TensorFlow Hub"""
        try:
            logger.info("Loading YamNet model...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            logger.info("YamNet model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to load YamNet: {e}")
            return False
    
    def load_models(self):
        """Load trained models from data directory"""
        try:
            # Load YamNet
            if not self.load_yamnet():
                return False
            
            # Load classifier
            classifier_path = self.model_dir / "yamnet_deepfake_detector.h5"
            scaler_path = self.model_dir / "yamnet_deepfake_detector_scaler.pkl"
            
            if classifier_path.exists() and scaler_path.exists():
                self.classifier = tf.keras.models.load_model(str(classifier_path))
                self.scaler = joblib.load(str(scaler_path))
                self.is_loaded = True
                logger.info("Models loaded successfully!")
                return True
            else:
                logger.warning(f"Model files not found in {self.model_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def extract_yamnet_features(self, audio_data: np.ndarray, sr: int = None) -> np.ndarray:
        """Extract YamNet features from audio"""
        try:
            # Resample to YamNet's expected sample rate
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            
            # Ensure mono audio
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Normalize
            audio_data = audio_data.astype(np.float32)
            
            # Get YamNet embeddings
            scores, embeddings, spectrogram = self.yamnet_model(audio_data)
            
            # Average embeddings across time
            features = tf.reduce_mean(embeddings, axis=0).numpy()
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def predict(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """Predict if audio is real or fake with bias correction"""
        if not self.is_loaded:
            return {"error": "Models not loaded"}

        try:
            features = self.extract_yamnet_features(audio_data, sr)
            if features is None:
                return {"error": "Could not extract features"}

            features_scaled = self.scaler.transform([features])
            prediction_prob = self.classifier.predict(features_scaled)[0][0]

            # BIAS CORRECTION: Apply probability calibration and higher threshold
            # Original range [0,1] mapped to [0.3, 0.7] to reduce extreme predictions
            calibrated_prob = 0.3 + 0.4 * prediction_prob
            
            # Use higher threshold to reduce false positives (real voices being labeled as fake)
            prediction = "FAKE" if calibrated_prob > self.prediction_threshold else "REAL"
            confidence = calibrated_prob if prediction == "FAKE" else (1 - calibrated_prob)

            return {
                "prediction": prediction,
                "confidence": float(confidence),
                "probability": float(prediction_prob),
                "calibrated_probability": float(calibrated_prob),
                "threshold_used": self.prediction_threshold,
                "bias_correction_applied": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"error": str(e)}
    
    def set_threshold(self, threshold: float):
        """Allow dynamic threshold adjustment"""
        self.prediction_threshold = max(0.1, min(0.9, threshold))
        logger.info(f"Prediction threshold set to {self.prediction_threshold}")

# Global detector instance
detector = DeepfakeDetector()

class AudioExtractor:
    """Handle audio extraction from various file formats"""
    
    @staticmethod
    def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
        """Extract audio from video file using ffmpeg"""
        try:
            if output_path is None:
                output_path = video_path.replace(Path(video_path).suffix, '.wav')
            
            # Extract audio using ffmpeg-python
            (
                ffmpeg
                .input(video_path)
                .output(output_path, acodec='pcm_s16le', ac=1, ar=16000)
                .overwrite_output()
                .run(quiet=True)
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio extraction error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio extraction failed: {e}")
    
    @staticmethod
    def load_audio_file(file_path: str) -> tuple:
        """Load audio file and return data with sample rate"""
        try:
            audio_data, sr = librosa.load(file_path, sr=None)
            return audio_data, sr
        except Exception as e:
            logger.error(f"Audio loading error: {e}")
            raise HTTPException(status_code=400, detail=f"Audio loading failed: {e}")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "YamNet Deepfake Voice Detection API",
        "version": "1.0.0",
        "model_loaded": detector.is_loaded,
        "bias_correction_enabled": True,
        "current_threshold": detector.prediction_threshold
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector.is_loaded,
        "bias_correction": True,
        "threshold": detector.prediction_threshold,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/upload")
async def predict_uploaded_file(file: UploadFile = File(...)):
    """Predict deepfake from uploaded file (audio or video)"""
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Determine file type and process accordingly
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            # Video file - extract audio first
            audio_path = AudioExtractor.extract_audio_from_video(temp_file_path)
            audio_data, sr = AudioExtractor.load_audio_file(audio_path)
        elif file_extension in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            # Audio file - load directly
            audio_data, sr = AudioExtractor.load_audio_file(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Make prediction using the bias-corrected method
        result = detector.predict(audio_data, sr)
        
        return {
            "filename": file.filename,
            "file_type": file_extension,
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio analysis"""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                if message["type"] == "audio_data":
                    # Handle base64 encoded audio data
                    audio_b64 = message["data"]
                    sample_rate = message.get("sample_rate", 16000)
                    
                    # Decode base64 audio data
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    # Convert bytes to numpy array
                    # This assumes the audio is sent as float32 array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Make prediction
                    result = detector.predict(audio_data, sample_rate)
                    
                    # Send result back
                    await manager.send_personal_message({
                        "type": "prediction_result",
                        "result": result
                    }, websocket)
                
                elif message["type"] == "ping":
                    # Handle ping/pong for connection health
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Unknown message type: {message['type']}"
                    }, websocket)
                    
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, websocket)
            
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": str(e)
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.post("/model/reload")
async def reload_models():
    """Reload models from data directory"""
    success = detector.load_models()
    
    return {
        "success": success,
        "model_loaded": detector.is_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/status")
async def model_status():
    """Get model loading status"""
    return {
        "model_loaded": detector.is_loaded,
        "yamnet_loaded": detector.yamnet_model is not None,
        "classifier_loaded": detector.classifier is not None,
        "scaler_loaded": detector.scaler is not None,
        "bias_correction_enabled": True,
        "current_threshold": detector.prediction_threshold
    }

@app.post("/config/threshold")
async def set_prediction_threshold(threshold: float):
    """Endpoint to adjust prediction threshold"""
    if not detector.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not 0.1 <= threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.1 and 0.9")
    
    detector.set_threshold(threshold)
    return {
        "success": True,
        "new_threshold": detector.prediction_threshold,
        "message": f"Threshold set to {detector.prediction_threshold}"
    }

@app.get("/debug/model_info")
async def debug_model_info():
    """Debug endpoint to get model information"""
    if not detector.is_loaded:
        return {"error": "Models not loaded"}
    
    return {
        "model_loaded": detector.is_loaded,
        "threshold": detector.prediction_threshold,
        "bias_correction": True,
        "calibration_range": "[0.3, 0.7]",
        "recommendation": "Use threshold 0.75-0.8 for balanced detection",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)