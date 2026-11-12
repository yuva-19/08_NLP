from pathlib import Path
from datetime import datetime
import uuid

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from retinaface import RetinaFace
from deepface import DeepFace

from src.model import FaceRecognitionModel


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "latest_checkpoint.pth"
REGISTERED_FACES_DIR = PROJECT_ROOT / "registered_faces"

INPUT_SIZE = 112
EMBEDDING_SIZE = 128
FACE_THRESHOLD = 0.9489252368609133  # Temporary demo threshold from exploratory test pairs.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

POSES = [
    "Look straight",
    "Turn slightly left",
    "Turn slightly right",
    "Look slightly upward",
    "Look slightly downward",
    "Natural alternate angle",
]


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

@st.cache_resource
def load_recognition_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model_state = checkpoint["model_state_dict"]
    num_classes = model_state["arcface.weight"].shape[0]

    model = FaceRecognitionModel(
        embedding_size=EMBEDDING_SIZE,
        num_classes=num_classes
    ).to(DEVICE)

    model.load_state_dict(model_state, strict=True)
    model.eval()
    model.backbone.eval()

    return model.backbone


# ------------------------------------------------------------
# Face detection, anti-spoofing and embedding
# ------------------------------------------------------------

def detect_and_align(frame_bgr):
    detections = RetinaFace.detect_faces(frame_bgr)

    if not isinstance(detections, dict) or len(detections) == 0:
        return [], []

    aligned_faces = RetinaFace.extract_faces(
        img_path=frame_bgr,
        align=True
    )

    return list(detections.values()), aligned_faces


def check_liveness(frame_bgr):
    results = DeepFace.extract_faces(
        img_path=frame_bgr,
        detector_backend="retinaface",
        enforce_detection=True,
        align=True,
        anti_spoofing=True
    )

    if len(results) != 1:
        return False, 0.0, "Exactly one face is required"

    is_real = bool(results[0].get("is_real", False))
    score = float(results[0].get("antispoof_score", 0.0))

    if not is_real:
        return False, score, "Spoof suspected"

    return True, score, "Face passed anti-spoofing check"


def extract_embedding(aligned_face, backbone):
    face = np.asarray(aligned_face)

    # RetinaFace aligned crops are RGB. Convert to uint8 if returned as floats.
    if face.dtype != np.uint8:
        if face.max() <= 1.0:
            face = face * 255.0
        face = np.clip(face, 0, 255).astype(np.uint8)

    face = Image.fromarray(face).convert("RGB")
    face = face.resize((INPUT_SIZE, INPUT_SIZE))

    face = np.asarray(face, dtype=np.float32) / 255.0
    face = (face - 0.5) / 0.5

    face_tensor = torch.from_numpy(face)
    face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
    face_tensor = face_tensor.to(DEVICE)

    with torch.inference_mode():
        embedding = backbone(face_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def process_face(frame_bgr, backbone):
    detections, aligned_faces = detect_and_align(frame_bgr)

    if len(detections) == 0:
        return None, "No face detected", None

    if len(detections) > 1:
        return None, "Multiple faces detected. Show only one face.", None

    if len(aligned_faces) == 0:
        return None, "Face alignment failed", None

    is_real, spoof_score, message = check_liveness(frame_bgr)

    if not is_real:
        return None, message, spoof_score

    embedding = extract_embedding(aligned_faces[0], backbone)
    return embedding, "Face processed successfully", spoof_score


# ------------------------------------------------------------
# Registered-user storage and matching
# ------------------------------------------------------------

def load_registered_users():
    REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)
    users = []

    for file_path in REGISTERED_FACES_DIR.glob("*.npz"):
        try:
            with np.load(file_path, allow_pickle=False) as data:
                users.append({
                    "user_id": str(data["user_id"].item()),
                    "name": str(data["name"].item()),
                    "registered_at": str(data["registered_at"].item()),
                    "embeddings": data["embeddings"].astype(np.float32)
                })
        except Exception:
            continue

    return users


def save_registered_user(name, embeddings):
    user_id = str(uuid.uuid4())
    REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        REGISTERED_FACES_DIR / f"{user_id}.npz",
        user_id=np.array(user_id),
        name=np.array(name),
        registered_at=np.array(datetime.now().isoformat(timespec="seconds")),
        embeddings=np.asarray(embeddings, dtype=np.float32)
    )

    return user_id


def match_embedding(query_embedding):
    users = load_registered_users()

    if not users:
        return None, 0.0

    query_embedding = F.normalize(
        torch.tensor(query_embedding, dtype=torch.float32),
        p=2,
        dim=0
    ).numpy()

    best_user = None
    best_similarity = -1.0

    for user in users:
        templates = F.normalize(
            torch.tensor(user["embeddings"], dtype=torch.float32),
            p=2,
            dim=1
        ).numpy()

        similarity = float(np.max(templates @ query_embedding))

        if similarity > best_similarity:
            best_similarity = similarity
            best_user = user

    if best_similarity >= FACE_THRESHOLD:
        return best_user, best_similarity

    return None, best_similarity


def image_to_bgr(uploaded_image):
    image = Image.open(uploaded_image).convert("RGB")
    image_array = np.asarray(image)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------

st.set_page_config(
    page_title="Face Authentication",
    page_icon="🔐",
    layout="centered"
)

st.markdown(
    """
    <style>
    .block-container {max-width: 850px; padding-top: 2rem;}
    .app-subtitle {color: #8b8b8b; margin-top: -12px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔐 Face Authentication")
st.markdown(
    '<p class="app-subtitle">MobileFaceNet · RetinaFace · FasNet</p>',
    unsafe_allow_html=True
)

try:
    backbone = load_recognition_model()
except Exception as error:
    st.error(f"Could not load the recognition model: {error}")
    st.stop()

REGISTERED_FACES_DIR.mkdir(parents=True, exist_ok=True)

page = st.sidebar.radio(
    "Menu",
    ["Authenticate", "Register New User", "Manage Users"]
)

st.sidebar.caption(f"Device: {DEVICE.type.upper()}")
st.sidebar.caption(f"Registered users: {len(load_registered_users())}")


# ------------------------------------------------------------
# Authenticate
# ------------------------------------------------------------

if page == "Authenticate":
    st.subheader("Authenticate")
    st.write("Position your face in the camera and capture one image.")

    if "auth_camera_version" not in st.session_state:
        st.session_state.auth_camera_version = 0

    camera_key = f"auth_camera_{st.session_state.auth_camera_version}"
    photo = st.camera_input("Capture your face", key=camera_key)

    if photo is not None:
        if st.button("Verify Face", type="primary", use_container_width=True):
            with st.spinner("Checking face and matching identity..."):
                try:
                    frame = image_to_bgr(photo)
                    embedding, message, spoof_score = process_face(frame, backbone)

                    if embedding is None:
                        st.error(message)
                    else:
                        user, similarity = match_embedding(embedding)

                        if user is None and len(load_registered_users()) == 0:
                            st.warning("No registered users. Register a user first.")
                        elif user is not None:
                            st.success("ACCESS GRANTED")
                            st.metric("Cosine similarity", f"{similarity:.4f}")
                            st.markdown(f"### Welcome, {user['name']}!")
                        else:
                            st.error("ACCESS DENIED")
                            st.metric("Best similarity", f"{similarity:.4f}")

                        if spoof_score is not None:
                            st.caption(f"Anti-spoofing score: {spoof_score:.4f}")

                except Exception as error:
                    st.error(f"Authentication failed: {error}")

            # No camera loop is running here. Streamlit processes this single
            # snapshot and leaves the result visible. Use Retake for another attempt.

    st.caption(
        "Each attempt processes one captured frame and then stops automatically. "
        "Use Retake in the camera widget to start another attempt."
    )


# ------------------------------------------------------------
# Register new user
# ------------------------------------------------------------

elif page == "Register New User":
    st.subheader("Register New User")
    st.write("Capture six face samples. Follow the pose instruction for each capture.")

    if "registration_active" not in st.session_state:
        st.session_state.registration_active = False
    if "registration_name" not in st.session_state:
        st.session_state.registration_name = ""
    if "registration_embeddings" not in st.session_state:
        st.session_state.registration_embeddings = []
    if "registration_pose_index" not in st.session_state:
        st.session_state.registration_pose_index = 0
    if "registration_camera_version" not in st.session_state:
        st.session_state.registration_camera_version = 0
    if "registration_complete_message" not in st.session_state:
        st.session_state.registration_complete_message = ""

    if st.session_state.registration_complete_message:
        st.success(st.session_state.registration_complete_message)
        st.session_state.registration_complete_message = ""

    if not st.session_state.registration_active:
        name = st.text_input("Full name")

        if st.button("Start Registration", type="primary", use_container_width=True):
            if not name.strip():
                st.warning("Enter a name to continue.")
            else:
                st.session_state.registration_active = True
                st.session_state.registration_name = name.strip()
                st.session_state.registration_embeddings = []
                st.session_state.registration_pose_index = 0
                st.session_state.registration_camera_version = 0
                st.rerun()

    else:
        pose_index = st.session_state.registration_pose_index
        embeddings = st.session_state.registration_embeddings

        st.progress(len(embeddings) / len(POSES))
        st.markdown(f"**Capture {pose_index + 1} of 6:** {POSES[pose_index]}")

        camera_key = (
            f"registration_camera_{st.session_state.registration_camera_version}"
        )
        photo = st.camera_input("Capture pose", key=camera_key)

        col1, col2 = st.columns(2)

        with col1:
            if photo is not None:
                if st.button("Process & Save Pose", type="primary", use_container_width=True):
                    with st.spinner("Processing face..."):
                        try:
                            frame = image_to_bgr(photo)
                            embedding, message, spoof_score = process_face(frame, backbone)

                            if embedding is None:
                                st.error(message)
                            else:
                                st.session_state.registration_embeddings.append(embedding)
                                st.session_state.registration_pose_index += 1
                                st.session_state.registration_camera_version += 1
                                st.success(
                                    f"Pose saved ({len(st.session_state.registration_embeddings)}/6)"
                                )

                                if len(st.session_state.registration_embeddings) == 6:
                                    user_id = save_registered_user(
                                        st.session_state.registration_name,
                                        st.session_state.registration_embeddings
                                    )
                                    st.session_state.registration_active = False
                                    st.session_state.registration_embeddings = []
                                    st.session_state.registration_pose_index = 0
                                    st.session_state.registration_complete_message = (
                                        f"Registration complete! User ID: {user_id}"
                                    )
                                st.rerun()

                        except Exception as error:
                            st.error(f"Could not process this capture: {error}")

        with col2:
            if st.button("Cancel Registration", use_container_width=True):
                st.session_state.registration_active = False
                st.session_state.registration_embeddings = []
                st.session_state.registration_pose_index = 0
                st.rerun()

        st.caption(
            "Only face embeddings and registration metadata are stored. "
            "The captured photographs are not saved to disk."
        )


# ------------------------------------------------------------
# Manage users
# ------------------------------------------------------------

elif page == "Manage Users":
    st.subheader("Manage Registered Users")

    users = load_registered_users()

    search_text = st.text_input("Search by name or user ID")

    filtered_users = [
        user for user in users
        if search_text.lower() in user["name"].lower()
        or search_text.lower() in user["user_id"].lower()
    ]

    if not filtered_users:
        st.info("No registered users found.")
    else:
        for user in filtered_users:
            with st.container(border=True):
                st.markdown(f"**{user['name']}**")
                st.caption(f"ID: {user['user_id']}")
                st.caption(f"Registered: {user['registered_at']}")
                st.caption(f"Face templates: {len(user['embeddings'])}")

        st.divider()
        st.markdown("**Delete a registered user**")

        delete_options = {
            f"{user['name']} — {user['user_id']}": user["user_id"]
            for user in filtered_users
        }

        selected_label = st.selectbox(
            "Select user",
            list(delete_options.keys())
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to permanently delete this user's face embeddings."
        )

        if st.button("Delete User", disabled=not confirm_delete):
            selected_id = delete_options[selected_label]
            file_path = REGISTERED_FACES_DIR / f"{selected_id}.npz"

            if file_path.exists():
                file_path.unlink()
                st.success("User deleted successfully.")
                st.rerun()
            else:
                st.error("User record was not found.")

        st.caption(
            "Deleting a user removes only that user's registered embeddings. "
            "The trained model and checkpoints are never deleted."
        )


st.divider()
st.caption(
    "Educational face-authentication demo. Anti-spoofing is not a guarantee "
    "against every presentation attack. The configured similarity threshold "
    "is exploratory and must be calibrated on validation data before real-world use."
)
