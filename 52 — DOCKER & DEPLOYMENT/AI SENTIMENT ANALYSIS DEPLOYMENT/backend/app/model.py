import torch
from transformers import pipeline


MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


class SentimentModel:

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1

        print("Loading sentiment analysis model...")
        print(f"Device: {'GPU' if self.device == 0 else 'CPU'}")

        self.classifier = pipeline(
            task="sentiment-analysis",
            model=MODEL_ID,
            device=self.device
        )

        print("Sentiment model loaded successfully!")

    def predict(self, text: str) -> dict:

        result = self.classifier(text)[0]

        return {
            "label": result["label"],
            "confidence": float(result["score"])
        }