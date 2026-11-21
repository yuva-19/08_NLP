from pydantic import BaseModel, Field


# ==========================================
# NLP REQUEST SCHEMA
# ==========================================

class TextRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text input for sentiment analysis"
    )


# ==========================================
# NLP RESPONSE SCHEMA
# ==========================================

class SentimentResponse(BaseModel):
    label: str = Field(
        ...,
        description="Predicted sentiment label"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score"
    )


# ==========================================
# CV RESPONSE SCHEMA
# ==========================================

class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionResponse(BaseModel):
    total_objects: int
    detections: list[Detection]