from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .model import SentimentModel
from .schemas import SentimentRequest, SentimentResponse


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting application...")
    print("Loading AI model...")

    app.state.sentiment_model = SentimentModel()

    print("Application startup complete!")

    yield

    print("Shutting down application...")


app = FastAPI(
    title="AI Sentiment Analysis API",
    description="REST API for sentiment analysis using DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "AI Sentiment Analysis API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():

    model_loaded = hasattr(app.state, "sentiment_model")

    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


# --------------------------------------------------
# SENTIMENT PREDICTION ENDPOINT
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=SentimentResponse
)
def predict_sentiment(request: SentimentRequest):

    try:

        result = app.state.sentiment_model.predict(
            request.text
        )

        return SentimentResponse(
            label=result["label"],
            confidence=result["confidence"]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )