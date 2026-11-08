from pathlib import Path
from urllib.parse import unquote

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ---------------------------------------------------------
# 1. Project paths and configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_feature_extraction.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# 2. Image preprocessing
#    Must match the preprocessing used during evaluation.
# ---------------------------------------------------------

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# 3. Load the trained ResNet50 model and checkpoint
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Place best_feature_extraction.pth inside the checkpoints folder."
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    category_to_label = checkpoint["category_to_label"]
    label_to_category = checkpoint["label_to_category"]
    num_classes = len(category_to_label)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, label_to_category, num_classes


# ---------------------------------------------------------
# 4. Generate Top-5 predictions
# ---------------------------------------------------------

def predict_image(image, model, label_to_category, num_classes):
    image = image.convert("RGB")
    image_tensor = test_transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        top_k = min(5, num_classes)
        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=top_k
        )

    predictions = []

    for probability, class_index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist()
    ):
        category = label_to_category.get(
            class_index,
            label_to_category.get(str(class_index), f"Class {class_index}")
        )
        category = unquote(str(category)).replace("_", " ")

        predictions.append({
            "landmark": category,
            "confidence": probability * 100
        })

    return predictions


# ---------------------------------------------------------
# 5. Streamlit interface
# ---------------------------------------------------------

st.set_page_config(
    page_title="Landmark Recognition",
    page_icon="🏛️",
    layout="centered"
)

st.title("🏛️ Landmark Recognition")
st.write(
    "Upload an image to identify its most likely class among "
    "the 51 landmark classes."
)

st.caption(f"Device: {DEVICE}")

uploaded_file = st.file_uploader(
    "Upload a landmark image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")
    st.image(image, use_container_width=True)

    try:
        with st.spinner("Predicting landmark..."):
            model, label_to_category, num_classes = load_model()
            predictions = predict_image(
                image,
                model,
                label_to_category,
                num_classes
            )

        best_prediction = predictions[0]

        st.subheader("Prediction")
        st.success(f"**{best_prediction['landmark']}**")
        st.metric(
            "Top prediction probability",
            f"{best_prediction['confidence']:.2f}%"
        )

        st.subheader("Top 5 probabilities")

        for rank, prediction in enumerate(predictions, start=1):
            st.write(
                f"**{rank}. {prediction['landmark']}** — "
                f"{prediction['confidence']:.2f}%"
            )
            st.progress(
                min(max(prediction["confidence"] / 100, 0.0), 1.0)
            )

        st.info(
            "These are softmax probabilities from the model. "
            "They represent relative model confidence and do not "
            "guarantee that a prediction is correct."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
