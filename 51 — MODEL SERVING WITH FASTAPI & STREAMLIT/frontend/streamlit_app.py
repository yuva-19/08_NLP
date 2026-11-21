import requests
import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Model Serving",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# 2. CONFIGURATION
# ==========================================

API_URL = "http://127.0.0.1:8000"

st.title("🤖 AI Model Serving Platform")

st.caption(
    "NLP Sentiment Analysis & Computer Vision "
    "powered by FastAPI"
)


# ==========================================
# 3. API STATUS
# ==========================================

st.sidebar.header("⚙️ Backend Status")

if st.sidebar.button("Check API Status"):

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=10
        )

        if response.status_code == 200:

            health = response.json()

            st.sidebar.success("API Online 🟢")

            st.sidebar.json(health)

        else:

            st.sidebar.error("API returned an error")

    except requests.exceptions.RequestException:

        st.sidebar.error(
            "API unavailable. Start the FastAPI server."
        )


# ==========================================
# 4. TABS
# ==========================================

nlp_tab, cv_tab = st.tabs(
    [
        "🧠 Sentiment Analysis",
        "👁️ Object Detection"
    ]
)


# ==========================================
# 5. NLP APPLICATION
# ==========================================

with nlp_tab:

    st.header("🧠 Sentiment Analysis")

    st.write(
        "Analyze text and predict whether the sentiment "
        "is positive or negative."
    )

    text_input = st.text_area(
        "Enter text",
        placeholder="I absolutely love this product!",
        height=150,
        max_chars=5000
    )

    if st.button(
        "Analyze Sentiment",
        type="primary",
        key="sentiment_button"
    ):

        if not text_input.strip():

            st.warning("Please enter some text.")

        else:

            try:

                with st.spinner("Analyzing text..."):

                    response = requests.post(
                        f"{API_URL}/predict",
                        json={
                            "text": text_input
                        },
                        timeout=60
                    )

                if response.status_code == 200:

                    result = response.json()

                    label = result["label"]
                    confidence = result["confidence"]

                    if label == "POSITIVE":

                        st.success(
                            f"Sentiment: {label} 😊"
                        )

                    else:

                        st.error(
                            f"Sentiment: {label} 😞"
                        )

                    st.metric(
                        "Confidence",
                        f"{confidence:.2%}"
                    )

                    st.progress(
                        min(max(confidence, 0.0), 1.0)
                    )

                else:

                    st.error(
                        f"API Error: {response.status_code}"
                    )

                    st.json(response.json())

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to API: {error}"
                )


# ==========================================
# 6. COMPUTER VISION APPLICATION
# ==========================================

with cv_tab:

    st.header("👁️ Object Detection")

    st.write(
        "Upload an image to detect objects using YOLO."
    )

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_image is not None:

        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Detect Objects",
            type="primary",
            key="detect_button"
        ):

            try:

                with st.spinner(
                    "Running object detection..."
                ):

                    response = requests.post(
                        f"{API_URL}/detect",
                        files={
                            "file": (
                                uploaded_image.name,
                                uploaded_image.getvalue(),
                                uploaded_image.type
                            )
                        },
                        timeout=120
                    )

                if response.status_code == 200:

                    result = response.json()

                    st.success("Detection completed! 🚀")

                    st.metric(
                        "Objects Detected",
                        result["total_objects"]
                    )

                    detections = result["detections"]

                    if detections:

                        st.subheader("Detection Results")

                        st.dataframe(
                            detections,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "No objects detected in this image."
                        )

                else:

                    st.error(
                        f"Detection failed: {response.status_code}"
                    )

                    st.json(response.json())

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to API: {error}"
                )


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Model Serving Project | "
    "FastAPI + Streamlit + Transformers + YOLO"
)