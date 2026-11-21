from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text input for sentiment analysis"
    )


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