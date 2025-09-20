import requests
import json

class DeepfakeDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_file(self, file_path):
        """Upload file for deepfake detection"""
        try:
            with open(file_path, "rb") as file:
                files = {"file": file}
                response = requests.post(f"{self.base_url}/predict/upload", files=files)
                
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Request failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def model_status(self):
        """Get model loading status"""
        response = requests.get(f"{self.base_url}/model/status")
        return response.json()

# Example usage
if __name__ == "__main__":
    client = DeepfakeDetectionClient()
    
    # Check health
    health = client.health_check()
    print("Health:", health)
    
    # Check model status
    status = client.model_status()
    print("Model Status:", status)
    