
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECOGNITION_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "epoch_checkpoint.pth"
)

ANTISPOOF_CHECKPOINT = (
    PROJECT_ROOT / "models" / "anti_spoofing" / "MiniFASNetV2.pth"
)

ANTISPOOF_DIR = (
    PROJECT_ROOT / "third_party" / "face-anti-spoofing"
)

# Import third-party MiniFASNetV2 architecture
sys.path.insert(0, str(ANTISPOOF_DIR))

from models.fastnet import MiniFASNetV2

from uniface import RetinaFace
from src.face_recognition_model import FaceRecognitionModel
from src.transforms import get_eval_transforms


class FaceEngine:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # MobileFaceNet + ArcFace model
        self.recognition_model = FaceRecognitionModel(
            embedding_size=128,
            num_classes=5103
        ).to(self.device)

        recognition_checkpoint = torch.load(
            RECOGNITION_CHECKPOINT,
            map_location=self.device
        )

        self.recognition_model.load_state_dict(
            recognition_checkpoint["model_state_dict"]
        )

        self.recognition_model.eval()

        # MiniFASNetV2 anti-spoofing model
        self.antispoof_model = MiniFASNetV2().to(self.device)

        antispoof_checkpoint = torch.load(
            ANTISPOOF_CHECKPOINT,
            map_location=self.device
        )

        self.antispoof_model.load_state_dict(
            antispoof_checkpoint
        )

        self.antispoof_model.eval()

        # RetinaFace detector
        self.face_detector = RetinaFace()

        # MobileFaceNet evaluation preprocessing
        self.recognition_transform = get_eval_transforms()

        print("Face Engine initialized successfully.")
        print("Device:", self.device)

    # --------------------------------------------------
    # FACE ANALYSIS
    # RetinaFace + MiniFASNetV2 + MobileFaceNet
    # --------------------------------------------------

    def analyze_frame(self, image):

        results = []

        if image is None or image.size == 0:
            raise ValueError("Invalid input image.")

        detected_faces = self.face_detector.detect(image)

        if detected_faces is None or len(detected_faces) == 0:
            return results

        with torch.inference_mode():

            for face in detected_faces:

                bbox = np.asarray(
                    face.bbox,
                    dtype=np.float32
                )

                detection_confidence = float(face.confidence)

                x1, y1, x2, y2 = bbox.astype(int)

                # Expand bounding box by 2.7
                width = x2 - x1
                height = y2 - y1

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                crop_width = width * 2.7
                crop_height = height * 2.7

                left = max(
                    0,
                    int(center_x - crop_width / 2)
                )

                top = max(
                    0,
                    int(center_y - crop_height / 2)
                )

                right = min(
                    image.shape[1],
                    int(center_x + crop_width / 2)
                )

                bottom = min(
                    image.shape[0],
                    int(center_y + crop_height / 2)
                )

                face_crop = image[top:bottom, left:right]

                if face_crop.size == 0:
                    continue

                # MiniFASNetV2 preprocessing
                # BGR, 80x80, float32, CHW
                antispoof_image = cv2.resize(
                    face_crop,
                    (80, 80)
                )

                antispoof_image = (
                    antispoof_image.astype(np.float32)
                    .transpose(2, 0, 1)
                )

                antispoof_tensor = torch.from_numpy(
                    antispoof_image
                ).unsqueeze(0).to(self.device)

                # Anti-spoofing inference
                antispoof_logits = self.antispoof_model(
                    antispoof_tensor
                )

                antispoof_probabilities = torch.softmax(
                    antispoof_logits,
                    dim=1
                )[0]

                antispoof_class = int(
                    torch.argmax(
                        antispoof_probabilities
                    ).item()
                )

                antispoof_confidence = float(
                    antispoof_probabilities[
                        antispoof_class
                    ].item()
                )

                # Repository class mapping:
                # Class 1 = Real
                # Classes 0 and 2 = Fake
                is_real = antispoof_class == 1

                result = {
                    "bbox": bbox.tolist(),
                    "detection_confidence": detection_confidence,
                    "is_real": is_real,
                    "antispoof_class": antispoof_class,
                    "antispoof_confidence": antispoof_confidence,
                    "antispoof_probabilities": (
                        antispoof_probabilities.cpu().tolist()
                    ),
                    "embedding": None
                }

                # Only genuine faces proceed to recognition
                if is_real:

                    # BGR -> RGB
                    face_rgb = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2RGB
                    )

                    face_pil = Image.fromarray(face_rgb)

                    recognition_tensor = (
                        self.recognition_transform(face_pil)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    # Extract 128D embedding
                    embedding = self.recognition_model.backbone(
                        recognition_tensor
                    )

                    embedding = F.normalize(
                        embedding,
                        p=2,
                        dim=1
                    )

                    result["embedding"] = (
                        embedding.squeeze(0).cpu()
                    )

                results.append(result)

        return results

    # --------------------------------------------------
    # FACE REGISTRATION
    # --------------------------------------------------

    def register_identity(
        self,
        identity_name,
        images,
        face_database
    ):

        if not identity_name or not identity_name.strip():
            raise ValueError("Identity name cannot be empty.")

        if identity_name in face_database:
            raise ValueError(
                f"{identity_name} is already registered."
            )

        if not images:
            raise ValueError("No registration images provided.")

        accepted_embeddings = []
        rejected_images = []

        for index, image in enumerate(images):

            results = self.analyze_frame(image)

            if len(results) != 1:
                rejected_images.append({
                    "image": index + 1,
                    "reason": "Expected exactly one face."
                })
                continue

            result = results[0]

            if not result["is_real"]:
                rejected_images.append({
                    "image": index + 1,
                    "reason": "Anti-spoofing classified it as fake."
                })
                continue

            accepted_embeddings.append(
                result["embedding"].clone()
            )

        if not accepted_embeddings:

            return {
                "success": False,
                "identity": identity_name,
                "accepted": 0,
                "rejected": rejected_images
            }

        face_database[identity_name] = {
            "embeddings": accepted_embeddings,
            "num_embeddings": len(accepted_embeddings)
        }

        return {
            "success": True,
            "identity": identity_name,
            "accepted": len(accepted_embeddings),
            "rejected": rejected_images
        }

    # --------------------------------------------------
    # IDENTITY AUTHENTICATION
    # --------------------------------------------------

    def authenticate_identity(
        self,
        embedding,
        face_database,
        threshold=0.9491418
    ):

        if embedding is None or not face_database:
            return {
                "authenticated": False,
                "identity": None,
                "similarity": None
            }

        query_embedding = F.normalize(
            embedding.reshape(1, -1),
            p=2,
            dim=1
        )

        best_identity = None
        best_similarity = -1.0

        for identity, data in face_database.items():

            registered_embeddings = torch.stack(
                data["embeddings"]
            ).to(embedding.device)

            registered_embeddings = F.normalize(
                registered_embeddings,
                p=2,
                dim=1
            )

            similarities = F.cosine_similarity(
                query_embedding,
                registered_embeddings,
                dim=1
            )

            # Average similarity across all registered templates
            identity_similarity = similarities.max().item()

            if identity_similarity > best_similarity:

                best_similarity = identity_similarity
                best_identity = identity

        authenticated = best_similarity >= threshold

        return {
            "authenticated": authenticated,
            "identity": best_identity if authenticated else None,
            "similarity": best_similarity
        }

    # --------------------------------------------------
    # COMBINED ACCESS DECISION
    # --------------------------------------------------

    def make_access_decision(
        self,
        analysis_results,
        face_database,
        threshold=0.9491418
    ):

        if len(analysis_results) == 0:
            return {
                "access_granted": False,
                "identity": None,
                "reason": "No face detected."
            }

        if len(analysis_results) > 1:
            return {
                "access_granted": False,
                "identity": None,
                "reason": "Multiple faces detected."
            }

        face = analysis_results[0]

        if not face["is_real"]:
            return {
                "access_granted": False,
                "identity": None,
                "reason": "Fake face detected."
            }

        auth_result = self.authenticate_identity(
            face["embedding"],
            face_database,
            threshold
        )

        if not auth_result["authenticated"]:
            return {
                "access_granted": False,
                "identity": None,
                "reason": "Identity not recognized.",
                "similarity": auth_result["similarity"]
            }

        return {
            "access_granted": True,
            "identity": auth_result["identity"],
            "reason": "Face verified successfully.",
            "similarity": auth_result["similarity"]
        }
