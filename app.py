import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List
from facenet_pytorch import InceptionResnetV1
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from facenet_pytorch import InceptionResnetV1
import torch.optim as optim
import torch
from PIL import Image as PILImage
from scipy.spatial.distance import euclidean
from models import MachineKernel
import os
from models import FaceRecognitionModel
from PIL import Image
import io
import json
from io import BytesIO
import cv2
import base64

# Defining Face Recognition Base Model


classes = np.load('train_dataset_classes.npy')  # Load up classes
print("Classes", classes)

train_embeddings = np.load('train_embeddings.npy')
train_labels = np.load('train_labels.npy')

# Load the model (ensure you are on the right device - CPU or GPU)
model = FaceRecognitionModel(len(classes))
model.eval()

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
model_path = os.path.join(script_dir, "face_recognition_model_full.pth")
print(model_path)

# loaded_model = torch.load(model_path)
# loaded_model = torch.load(model_path)
# loaded_model.eval()

model.load_state_dict(torch.load(model_path, map_location="cpu"))

ml_kernel = MachineKernel()

# Define FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the input data model (adjust according to your needs)


def generate_noise(image_array, epsilon, attack_type):
    """
    Generates adversarial noise based on the attack type.
    """
    noise = np.random.normal(0, 0.1, image_array.shape) * \
        255  # Default random noise

    if attack_type == "fgsm":
        noise = np.sign(np.random.randn(*image_array.shape)) * epsilon * 255
    elif attack_type == "pgd":
        noise = np.clip(np.random.randn(*image_array.shape)
                        * epsilon * 255, -30, 30)

    return noise.astype(np.uint8)


class InputData(BaseModel):
    attack_type: str = "fgsm"  # The attack type: "FGSM" or "PGD"
    # Epsilon for adversarial perturbation (default to 0.1)
    epsilon: float = 0.1


@app.post("/predict")
async def predict(input_data: str = Form(...), file: UploadFile = File(...)):
    input_dict = json.loads(input_data)
    input_data = InputData(**input_dict)  # Convert to Pydantic model

    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')

    # Make prediction on image
    prediction, confidence = ml_kernel.predict_single_image(
        image, model, train_embeddings, train_labels, input_data.attack_type, input_data.epsilon)

    print("GOT TO THIS FINAL STAGE")
    return {"prediction": prediction, "confidence": confidence.item()}


@app.post("/generate_noise")
async def generate_noise_api(file: UploadFile = File(...), epsilon: float = Form(...), attack_type: str = Form(...)):
    """
    API endpoint to generate adversarial noise for an image.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)

    noise = generate_noise(image_array, epsilon, attack_type)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)

    # Convert noise to Base64
    _, buffer = cv2.imencode(".png", noise)
    noise_base64 = base64.b64encode(buffer).decode("utf-8")

    # Convert noisy image to Base64
    _, noisy_image_buffer = cv2.imencode(".png", noisy_image)
    noisy_image_base64 = base64.b64encode(noisy_image_buffer).decode("utf-8")

    return {
        "noise_image": f"data:image/png;base64,{noise_base64}",
        "noisy_image": f"data:image/png;base64,{noisy_image_base64}"
    }
