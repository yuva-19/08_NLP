
import cv2
import tkinter as tk

from tkinter import ttk, messagebox

from src.face_engine import FaceEngine
from src.face_database import FaceDatabase


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

REGISTRATION_IMAGES = 6
CAPTURE_DELAY = 8
MATCH_THRESHOLD = 0.9491418


# --------------------------------------------------
# WEBCAM CAPTURE
# --------------------------------------------------

def capture_registration_images(num_images=6, delay=4):

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    captured_images = []

    try:
        print("Registration started. Adjust your face for each capture.")

        for i in range(num_images):

            for remaining in range(delay, 0, -1):

                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError("Unable to read webcam frame.")

                preview = frame.copy()

                cv2.putText(
                    preview,
                    f"Capture {i + 1}/{num_images}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    preview,
                    f"Capturing in: {remaining}",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3
                )

                cv2.imshow("Face Registration", preview)

                if cv2.waitKey(1000) & 0xFF == 27:
                    return []

            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Unable to capture image.")

            captured_images.append(frame.copy())

            print(f"Captured image {i + 1}/{num_images}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_images


def capture_authentication_image():

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    captured_image = None

    try:
        print("Press SPACE to capture your face. Press ESC to cancel.")

        while True:

            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Unable to read webcam frame.")

            preview = frame.copy()

            cv2.putText(
                preview,
                "Press SPACE to authenticate",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Face Authentication", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:
                captured_image = frame.copy()
                break

            elif key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured_image


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

class FaceAuthenticationApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Face Authentication System")
        self.root.geometry("650x750")
        self.root.resizable(True, True)

        # Initialize models
        self.engine = FaceEngine()

        # Initialize persistent database
        self.storage = FaceDatabase()
        self.face_database = self.storage.load_database()

        self.create_interface()
        self.refresh_identity_list()

    # --------------------------------------------------
    # INTERFACE
    # --------------------------------------------------

    def create_interface(self):

        main_frame = ttk.Frame(self.root, padding=25)
        main_frame.pack(fill="both", expand=True)

        ttk.Label(
            main_frame,
            text="FACE AUTHENTICATION SYSTEM",
            font=("Arial", 18, "bold")
        ).pack(pady=15)

        ttk.Label(
            main_frame,
            text="Face Registration and Identity Verification",
            font=("Arial", 10)
        ).pack(pady=5)

        # Registration section
        registration_frame = ttk.LabelFrame(
            main_frame,
            text="Register New Identity",
            padding=15
        )

        registration_frame.pack(fill="x", pady=15)

        self.name_entry = ttk.Entry(
            registration_frame,
            font=("Arial", 11)
        )

        self.name_entry.pack(
            fill="x",
            pady=8
        )

        self.name_entry.insert(0, "Enter identity name")

        ttk.Button(
            registration_frame,
            text="Register Face",
            command=self.register_face
        ).pack(pady=8)

        # Authentication section
        authentication_frame = ttk.LabelFrame(
            main_frame,
            text="Face Authentication",
            padding=15
        )

        authentication_frame.pack(fill="x", pady=10)

        ttk.Button(
            authentication_frame,
            text="Authenticate Using Webcam",
            command=self.authenticate_face
        ).pack(pady=8)

        # Result display
        self.status_label = ttk.Label(
            main_frame,
            text="System Ready",
            font=("Arial", 13, "bold")
        )

        self.status_label.pack(pady=15)

        # Registered identities
        identity_frame = ttk.LabelFrame(
            main_frame,
            text="Registered Identities",
            padding=10
        )

        identity_frame.pack(fill="both", expand=True, pady=10)

        self.identity_list = tk.Listbox(
            identity_frame,
            height=5,
            font=("Arial", 10)
        )

        self.identity_list.pack(
            fill="both",
            expand=True,
            pady=5
        )

        ttk.Button(
            identity_frame,
            text="Delete Selected Identity",
            command=self.delete_identity
        ).pack(pady=5)

    # --------------------------------------------------
    # REFRESH REGISTERED IDENTITIES
    # --------------------------------------------------

    def refresh_identity_list(self):

        self.identity_list.delete(0, tk.END)

        for identity in self.storage.list_identities():
            self.identity_list.insert(tk.END, identity)

    # --------------------------------------------------
    # REGISTER FACE
    # --------------------------------------------------

    def register_face(self):

        identity_name = self.name_entry.get().strip()

        if not identity_name or identity_name == "Enter identity name":
            messagebox.showwarning(
                "Registration",
                "Please enter an identity name."
            )
            return

        if identity_name in self.face_database:
            messagebox.showwarning(
                "Registration",
                "This identity is already registered."
            )
            return

        try:

            self.status_label.config(
                text="Capturing registration images...",
                foreground="blue"
            )

            self.root.update_idletasks()

            images = capture_registration_images(
                REGISTRATION_IMAGES,
                CAPTURE_DELAY
            )

            if not images:
                self.status_label.config(text="Registration cancelled.")
                return

            registration_result = self.engine.register_identity(
                identity_name,
                images,
                self.face_database
            )

            if not registration_result["success"]:

                self.status_label.config(
                    text="Registration failed.",
                    foreground="red"
                )

                messagebox.showerror(
                    "Registration Failed",
                    "No valid face embeddings were collected."
                )

                return

            # Persist the newly registered identity
            self.storage.save_identity(
                identity_name,
                self.face_database[identity_name]
            )

            self.refresh_identity_list()

            self.status_label.config(
                text=f"Registered: {identity_name}",
                foreground="green"
            )

            messagebox.showinfo(
                "Registration Successful",
                f"Identity: {identity_name}\n"
                f"Accepted images: {registration_result['accepted']}\n"
                f"Rejected images: {len(registration_result['rejected'])}"
            )

        except Exception as error:

            self.status_label.config(
                text="Registration error.",
                foreground="red"
            )

            messagebox.showerror(
                "Error",
                str(error)
            )

    # --------------------------------------------------
    # AUTHENTICATE FACE
    # --------------------------------------------------

    def authenticate_face(self):

        if not self.face_database:

            messagebox.showwarning(
                "Authentication",
                "No registered identities found."
            )
            return

        try:

            image = capture_authentication_image()

            if image is None:
                self.status_label.config(text="Authentication cancelled.")
                return

            analysis_results = self.engine.analyze_frame(image)

            access_result = self.engine.make_access_decision(
                analysis_results,
                self.face_database,
                threshold=MATCH_THRESHOLD
            )

            if access_result["access_granted"]:

                identity = access_result["identity"]
                similarity = access_result["similarity"]

                self.status_label.config(
                    text=f"ACCESS GRANTED: {identity}",
                    foreground="green"
                )

                messagebox.showinfo(
                    "Access Granted",
                    f"Identity: {identity}\n"
                    f"Cosine Similarity: {similarity:.4f}"
                )

            else:

                self.status_label.config(
                    text="ACCESS DENIED",
                    foreground="red"
                )

                messagebox.showwarning(
                    "Access Denied",
                    access_result["reason"]
                )

        except Exception as error:

            self.status_label.config(
                text="Authentication error.",
                foreground="red"
            )

            messagebox.showerror(
                "Error",
                str(error)
            )

    # --------------------------------------------------
    # DELETE IDENTITY
    # --------------------------------------------------

    def delete_identity(self):

        selection = self.identity_list.curselection()

        if not selection:
            messagebox.showwarning(
                "Delete Identity",
                "Please select an identity."
            )
            return

        identity_name = self.identity_list.get(selection[0])

        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {identity_name}?"
        )

        if not confirm:
            return

        try:

            self.storage.delete_identity(identity_name)

            self.face_database.pop(identity_name, None)

            self.refresh_identity_list()

            self.status_label.config(
                text=f"Deleted: {identity_name}",
                foreground="blue"
            )

        except Exception as error:

            messagebox.showerror(
                "Deletion Error",
                str(error)
            )


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    root = tk.Tk()

    app = FaceAuthenticationApp(root)

    root.mainloop()
