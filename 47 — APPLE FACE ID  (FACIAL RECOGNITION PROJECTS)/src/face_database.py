
import json
from pathlib import Path

import torch


# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = PROJECT_ROOT / "data" / "registered_faces"


class FaceDatabase:

    def __init__(self):

        DATABASE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        self.database_dir = DATABASE_DIR

    # --------------------------------------------------
    # VALIDATE IDENTITY NAME
    # --------------------------------------------------

    def _get_file_path(self, identity_name):

        identity_name = identity_name.strip()

        if not identity_name:
            raise ValueError("Identity name cannot be empty.")

        invalid_characters = '<>:"/\\|?*'

        if any(char in identity_name for char in invalid_characters):
            raise ValueError("Identity name contains invalid characters.")

        if identity_name in {".", ".."} or identity_name.endswith((" ", ".")):
            raise ValueError("Invalid identity name.")

        return self.database_dir / f"{identity_name}.json"

    # --------------------------------------------------
    # SAVE ONE IDENTITY
    # --------------------------------------------------

    def save_identity(self, identity_name, identity_data):

        file_path = self._get_file_path(identity_name)

        if file_path.exists():
            raise FileExistsError(
                f"{identity_name} is already stored."
            )

        embeddings = identity_data["embeddings"]

        if not embeddings:
            raise ValueError("No embeddings available to save.")

        embedding_values = [
            embedding.detach().cpu().reshape(-1).tolist()
            for embedding in embeddings
        ]

        if any(len(embedding) != 128 for embedding in embedding_values):
            raise ValueError("Every embedding must contain 128 values.")

        data = {
            "identity": identity_name,
            "embedding_size": 128,
            "num_embeddings": len(embedding_values),
            "embeddings": embedding_values
        }

        temporary_path = file_path.with_suffix(".tmp")

        try:
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            temporary_path.replace(file_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        print(f"Identity saved: {identity_name}")

    # --------------------------------------------------
    # LOAD ONE IDENTITY
    # --------------------------------------------------

    def load_identity(self, identity_name):

        file_path = self._get_file_path(identity_name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No registered identity found: {identity_name}"
            )

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        embeddings = data["embeddings"]

        if data["embedding_size"] != 128:
            raise ValueError("Invalid embedding dimension.")

        if data["num_embeddings"] != len(embeddings):
            raise ValueError("Embedding count mismatch.")

        if not embeddings or any(len(item) != 128 for item in embeddings):
            raise ValueError("Invalid stored embeddings.")

        return {
            "embeddings": [
                torch.tensor(item, dtype=torch.float32)
                for item in embeddings
            ],
            "num_embeddings": len(embeddings)
        }

    # --------------------------------------------------
    # LOAD ALL REGISTERED IDENTITIES
    # --------------------------------------------------

    def load_database(self):

        face_database = {}

        for file_path in self.database_dir.glob("*.json"):

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            identity_name = data["identity"]

            face_database[identity_name] = self.load_identity(
                identity_name
            )

        return face_database

    # --------------------------------------------------
    # LIST REGISTERED IDENTITIES
    # --------------------------------------------------

    def list_identities(self):

        return sorted(
            file_path.stem
            for file_path in self.database_dir.glob("*.json")
        )

    # --------------------------------------------------
    # DELETE ONE IDENTITY
    # --------------------------------------------------

    def delete_identity(self, identity_name):

        file_path = self._get_file_path(identity_name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"No registered identity found: {identity_name}"
            )

        file_path.unlink()

        print(f"Identity deleted: {identity_name}")
