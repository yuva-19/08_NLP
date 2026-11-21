
import streamlit as st
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(
    page_title="Tesla Autopilot - Object Detection",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Tesla Autopilot — Object Detection")
st.caption("YOLO11n fine-tuned on BDD100K")

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

DEFAULT_MODEL = (
    Path(__file__).parent
    / "outputs"
    / "yolo11n_bdd100k"
    / "weights"
    / "best.pt"
)

st.sidebar.header("Detection Settings")

model_path = st.sidebar.text_input(
    "Model checkpoint",
    str(DEFAULT_MODEL)
)

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.01, 1.00, 0.25, 0.01
)

image_size = st.sidebar.select_slider(
    "Image Size",
    options=[320, 416, 512, 640, 768, 960],
    value=640
)

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Detect Objects", type="primary", use_container_width=True):

        if not Path(model_path).is_file():
            st.error(f"Checkpoint not found: {model_path}")

        else:
            with st.spinner("Detecting objects..."):

                detector = load_model(model_path)

                results = detector.predict(
                    np.array(image),
                    conf=conf_threshold,
                    imgsz=image_size,
                    verbose=False
                )

                result = results[0]

                annotated = result.plot()
                annotated = annotated[:, :, ::-1]

            with right:
                st.subheader("Detection Result")
                st.image(annotated, use_container_width=True)

            boxes = result.boxes
            st.metric("Objects detected", len(boxes))

            if len(boxes) > 0:

                rows = []

                for box in boxes:
                    class_id = int(box.cls.item())

                    rows.append({
                        "Class": result.names[class_id],
                        "Confidence": round(float(box.conf.item()), 4),
                        "X1": round(float(box.xyxy[0][0].item()), 1),
                        "Y1": round(float(box.xyxy[0][1].item()), 1),
                        "X2": round(float(box.xyxy[0][2].item()), 1),
                        "Y2": round(float(box.xyxy[0][3].item()), 1),
                    })

                st.subheader("Detected Objects")
                st.dataframe(rows, use_container_width=True, hide_index=True)

                output_buffer = BytesIO()
                Image.fromarray(annotated).save(output_buffer, format="PNG")

                st.download_button(
                    "⬇️ Download Annotated Image",
                    data=output_buffer.getvalue(),
                    file_name="detected_image.png",
                    mime="image/png"
                )

            else:
                st.info("No objects detected. Try lowering the confidence threshold.")
