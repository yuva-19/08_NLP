from contextlib import asynccontextmanager
from io import BytesIO

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from transformers import pipeline
from ultralytics import YOLO

from .schemas import (
    TextRequest,
    SentimentResponse,
    DetectionResponse,
    Detection
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

NLP_MODEL_ID = (
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

CV_MODEL_ID = "yolo11n.pt"

DEVICE = 0 if torch.cuda.is_available() else -1

DEVICE_NAME = (
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)


# ==========================================
# 2. APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 50)
    print("Starting AI Model Serving API")
    print("=" * 50)

    print(f"Device: {DEVICE_NAME}")

    print("\nLoading NLP model...")

    app.state.sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model=NLP_MODEL_ID,
        device=DEVICE
    )

    app.state.yolo_model = None

    print("NLP model loaded successfully 🚀")

    yield

    print("\nShutting down AI API...")


# ==========================================
# 3. FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="AI Model Serving API",
    description="""
    AI inference API providing:

    - NLP Sentiment Analysis
    - Computer Vision Object Detection
    """,
    version="1.0.0",
    lifespan=lifespan
)


# ==========================================
# 4. HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Model Serving API",
        "status": "running",
        "device": DEVICE_NAME,
        "available_endpoints": [
            "/health",
            "/predict",
            "/detect",
            "/docs"
        ]
    }


# ==========================================
# 5. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "device": DEVICE_NAME,
        "nlp_model_loaded": hasattr(
            app.state,
            "sentiment_pipeline"
        ),
        "cv_model_loaded": app.state.yolo_model is not None
    }


# ==========================================
# 6. NLP SENTIMENT ANALYSIS
# ==========================================

@app.post(
    "/predict",
    response_model=SentimentResponse
)
def predict_sentiment(request: TextRequest):

    try:

        result = app.state.sentiment_pipeline(
            request.text
        )[0]

        return SentimentResponse(
            label=result["label"],
            confidence=float(result["score"])
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Sentiment prediction failed: {str(error)}"
        )


# ==========================================
# 7. LOAD YOLO MODEL
# ==========================================

def load_yolo_model():

    if app.state.yolo_model is None:

        print("Loading YOLO object detection model...")

        app.state.yolo_model = YOLO(CV_MODEL_ID)

        print("YOLO model loaded successfully 🚀")

    return app.state.yolo_model


# ==========================================
# 8. COMPUTER VISION OBJECT DETECTION
# ==========================================

@app.post(
    "/detect",
    response_model=DetectionResponse
)
async def detect_objects(
    file: UploadFile = File(...)
):

    # --------------------------------------
    # Validate file type
    # --------------------------------------

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:

        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG or WEBP image"
        )

    try:

        image_bytes = await file.read()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # --------------------------------------
    # Load CV model
    # --------------------------------------

    try:

        model = load_yolo_model()

        results = model.predict(
            source=image,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # ----------------------------------
        # Extract detections
        # ----------------------------------

        detections = []

        boxes = result.boxes

        for box in boxes:

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            confidence = float(
                box.conf[0].detach().cpu()
            )

            class_id = int(
                box.cls[0].detach().cpu()
            )

            class_name = model.names[class_id]

            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=confidence,
                    x1=coordinates[0],
                    y1=coordinates[1],
                    x2=coordinates[2],
                    y2=coordinates[3]
                )
            )

        return DetectionResponse(
            total_objects=len(detections),
            detections=detections
        )

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Object detection failed: {str(error)}"
        )