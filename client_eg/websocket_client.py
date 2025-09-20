import asyncio
import websockets
import json
import numpy as np
import base64
import librosa

class WebSocketClient:
    def __init__(self, uri="ws://localhost:8000/ws"):
        self.uri = uri
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket"""
        self.websocket = await websockets.connect(self.uri)
        print("Connected to WebSocket")
    
    async def send_audio_data(self, audio_data, sample_rate=16000):
        """Send audio data for real-time detection"""
        if self.websocket is None:
            await self.connect()
        
        # Convert audio to base64
        audio_bytes = audio_data.astype(np.float32).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        message = {
            "type": "audio_data",
            "data": audio_b64,
            "sample_rate": sample_rate
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def receive_results(self):
        """Listen for results from server"""
        if self.websocket is None:
            return None
            
        try:
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            print(f"Error receiving: {e}")
            return None
    
    async def send_ping(self):
        """Send ping to keep connection alive"""
        if self.websocket is None:
            await self.connect()
        
        message = {"type": "ping"}
        await self.websocket.send(json.dumps(message))
    
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()

# Example real-time audio processing
async def real_time_detection_example():
    client = WebSocketClient()
    
    try:
        await client.connect()
        
        # Load a test audio file
        audio_path = "test_files/sample_audio.wav"  # Replace with your file
        audio_data, sr = librosa.load(audio_path, sr=16000)
        
        # Split audio into chunks for real-time processing
        chunk_size = sr * 2  # 2-second chunks
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            if len(chunk) > sr * 0.5:  # Only process chunks > 0.5 seconds
                # Send chunk
                await client.send_audio_data(chunk, sr)
                
                # Receive result
                result = await client.receive_results()
                print(f"Chunk {i//chunk_size + 1}: {result}")
                
                # Small delay between chunks
                await asyncio.sleep(0.1)
    
    finally:
        await client.close()
