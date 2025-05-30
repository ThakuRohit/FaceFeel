from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os
import traceback
from keras.models import load_model

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.keras')
model = load_model(model_path)

# Emotion and emoji maps
emotion_dict = {
    0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprised"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
emoji_map = {
    0: os.path.join(BASE_DIR, "emojis", "angry.png"),
    1: os.path.join(BASE_DIR, "emojis", "disgusted.png"),
    2: os.path.join(BASE_DIR, "emojis", "fearful.png"),
    3: os.path.join(BASE_DIR, "emojis", "happy.png"),
    4: os.path.join(BASE_DIR, "emojis", "neutral.png"),
    5: os.path.join(BASE_DIR, "emojis", "sad.png"),
    6: os.path.join(BASE_DIR, "emojis", "surprised.png")
}

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str

# Load Haarcascade once
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.post("/process-image")
async def process_image(data: ImageData):
    try:
        header, encoded = data.image.split(",")
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi_gray, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi, verbose=0)
            emotion_index = int(np.argmax(preds))
            emotion_label = emotion_dict[emotion_index]

            emoji_path = emoji_map[emotion_index]
            emoji_img = Image.open(emoji_path).convert("RGBA")
            buffer = BytesIO()
            emoji_img.save(buffer, format="PNG")
            encoded_emoji = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return JSONResponse({
                "emotion": emotion_label,
                "emoji": f"data:image/png;base64,{encoded_emoji}"
            })

        raise HTTPException(status_code=400, detail="Face not processed")

    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
