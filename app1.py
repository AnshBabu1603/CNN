import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from collections import deque
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from tempfile import NamedTemporaryFile
import uvicorn

# FastAPI instance
app = FastAPI()

# Enable CORS (Allow frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[1] Device Selected: {device}", flush=True)

# Define transformations for video frames
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define Model Class
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        return self.model(x)

# Load trained model
def load_model(model_path):
    print("[3] Loading Trained Model...", flush=True)
    model = DeepfakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("    - Model Loaded Successfully!", flush=True)
    return model

model_path = r"best_deepfake_model.pth"
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
else:
    model = load_model(model_path)

# Video Prediction Function
def predict_video(video_path, model, frame_sample_rate=5):
    cap = cv2.VideoCapture(video_path)
    frame_scores = deque()

    if not cap.isOpened():
        return {"error": "Could not open video"}

    frame_count = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                input_tensor = test_transform(pil_image).unsqueeze(0).to(device)

                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0]
                fake_score = prob[0].item()
                frame_scores.append(fake_score)

            frame_count += 1

    cap.release()

    avg_fake_score = np.mean(frame_scores) if frame_scores else 0
    result = "FAKE" if avg_fake_score > 0.5 else "REAL"

    return {
        "prediction": result,
        "confidence": round(avg_fake_score if result == "FAKE" else 1 - avg_fake_score, 2)
    }

# API Endpoint for Video Upload (Renamed to /analyze)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        prediction = predict_video(temp_file_path, model)

        os.remove(temp_file_path)
        return prediction

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# Health check endpoint
@app.get("/")
def root():
    return {"message": "DefakeX Media Guardian API is live!"}

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # This binds to all available network interfaces
