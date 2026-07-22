import customtkinter as ctk
import numpy as np
import pandas as pd
import tensorflow as tf

from joblib import load
from tkinter import messagebox

# ==========================================================
# APPEARANCE
# ==========================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ==========================================================
# APPLICATION
# ==========================================================

class DiabetesApp:

    def __init__(self):

        # ---------------- Window ----------------

        self.app = ctk.CTk()

        self.app.title("Diabetes Risk Prediction System")

        self.app.geometry("1200x720")

        self.app.resizable(False, False)

        # ---------------- Header ----------------

        self.create_header()

        # ---------------- Main Layout ----------------

        self.create_main_layout()

        # ---------------- Left Panel ----------------

        self.create_patient_panel()

        # ---------------- Right Panel ----------------

        self.create_result_panel()

        # ---------------- Buttons ----------------

        self.create_buttons()
        # ==========================================
        # Load TensorFlow Lite Model
        # ==========================================

        self.load_model()

    # ======================================================
    # HEADER
    # ======================================================

    def create_header(self):

        header = ctk.CTkFrame(
            self.app,
            corner_radius=0,
            height=90
        )

        header.pack(
            fill="x"
        )

        title = ctk.CTkLabel(
            header,
            text="🩺 DIABETES RISK PREDICTION SYSTEM",
            font=("Arial", 30, "bold")
        )

        title.pack(
            pady=(15, 2)
        )

        subtitle = ctk.CTkLabel(
            header,
            text="Federated Learning  •  TensorFlow Lite  •  Raspberry Pi 4",
            font=("Arial", 16)
        )

        subtitle.pack()

    # ======================================================
    # MAIN LAYOUT
    # ======================================================

    def create_main_layout(self):

        self.main = ctk.CTkFrame(
            self.app
        )

        self.main.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=20
        )

        # Left Panel

        self.left = ctk.CTkFrame(
            self.main,
            width=470
        )

        self.left.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(10,5),
            pady=10
        )

        # Right Panel

        self.right = ctk.CTkFrame(
            self.main,
            width=470
        )

        self.right.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(5,10),
            pady=10
        )

    # ======================================================
    # PATIENT PANEL
    # ======================================================

    def create_patient_panel(self):

        heading = ctk.CTkLabel(

            self.left,

            text="Patient Information",

            font=("Arial",22,"bold")

        )

        heading.grid(
            row=0,
            column=0,
            columnspan=2,
            pady=20
        )

        labels = [

            "Gender",

            "Age",

            "Hypertension",

            "Heart Disease",

            "Smoking History",

            "BMI",

            "HbA1c Level",

            "Blood Glucose"

        ]

        for i,label in enumerate(labels):

            ctk.CTkLabel(

                self.left,

                text=label,

                font=("Arial",15)

            ).grid(

                row=i+1,

                column=0,

                sticky="w",

                padx=20,

                pady=12

            )

        # ===========================
        # Widgets
        # ===========================

        self.gender = ctk.CTkComboBox(

            self.left,

            values=[
                "Male",
                "Female"
            ],

            width=220

        )

        self.gender.grid(
            row=1,
            column=1,
            padx=20
        )

        self.age = ctk.CTkEntry(

            self.left,

            width=220,

            placeholder_text="Enter Age"

        )

        self.age.grid(
            row=2,
            column=1,
            padx=20
        )

        self.hypertension = ctk.CTkComboBox(

            self.left,

            values=[
                "No",
                "Yes"
            ],

            width=220

        )

        self.hypertension.grid(
            row=3,
            column=1,
            padx=20
        )

        self.heart = ctk.CTkComboBox(

            self.left,

            values=[
                "No",
                "Yes"
            ],

            width=220

        )

        self.heart.grid(
            row=4,
            column=1,
            padx=20
        )

        self.smoking = ctk.CTkComboBox(

            self.left,

            values=[

                "never",

                "No Info",

                "current",

                "former",

                "ever",

                "not current"

            ],

            width=220

        )

        self.smoking.grid(
            row=5,
            column=1,
            padx=20
        )

        self.bmi = ctk.CTkEntry(

            self.left,

            width=220,

            placeholder_text="BMI"

        )

        self.bmi.grid(
            row=6,
            column=1,
            padx=20
        )

        self.hba1c = ctk.CTkEntry(

            self.left,

            width=220,

            placeholder_text="HbA1c"

        )

        self.hba1c.grid(
            row=7,
            column=1,
            padx=20
        )

        self.glucose = ctk.CTkEntry(

            self.left,

            width=220,

            placeholder_text="Blood Glucose"

        )

        self.glucose.grid(
            row=8,
            column=1,
            padx=20
        )

    # ======================================================
    # RESULT PANEL
    # ======================================================

    def create_result_panel(self):

        heading = ctk.CTkLabel(
            self.right,
            text="Prediction Dashboard",
            font=("Arial",24,"bold")
        )

        heading.pack(pady=(20,10))

    # ==========================
    # Model Status Card
    # ==========================

        status_frame = ctk.CTkFrame(
            self.right,
            width=380,
            height=70
        )

        status_frame.pack(
            pady=10,
            padx=20,
            fill="x"
        )

        status_title = ctk.CTkLabel(
            status_frame,
            text="Model Status",
            font=("Arial",16,"bold")
        )

        status_title.pack(pady=(10,0))

        self.model_status = ctk.CTkLabel(
            status_frame,
            text="TensorFlow Lite Model : Not Loaded",
            text_color="orange",
            font=("Arial",15)
        )

        self.model_status.pack()

    # ==========================
    # Prediction Card
    # ==========================

        result_frame = ctk.CTkFrame(
            self.right
        )

        result_frame.pack(
            pady=20,
            padx=20,
            fill="x"
        )

        self.result_status = ctk.CTkLabel(
            result_frame,
            text="Waiting for Prediction",
            font=("Arial",20,"bold")
        )

        self.result_status.pack(
            pady=(20,10)
        )

        self.risk_score = ctk.CTkLabel(
            result_frame,
            text="Risk Score : --",
            font=("Arial",18)
        )

        self.risk_score.pack(pady=8)

        self.probability = ctk.CTkLabel(
            result_frame,
            text="Probability : -- %",
            font=("Arial",18)
        )

        self.probability.pack(pady=8)

        self.risk_level = ctk.CTkLabel(
            result_frame,
            text="Risk Level : --",
            font=("Arial",20,"bold")
        )

        self.risk_level.pack(pady=8)

        self.prediction = ctk.CTkLabel(
            result_frame,
            text="Prediction : --",
            font=("Arial",18)
        )

        self.prediction.pack(pady=8)

    # ==========================
    # Progress Bar
    # ==========================

        ctk.CTkLabel(
            result_frame,
            text="Risk Probability",
            font=("Arial",16)
        ).pack(pady=(15,5))

        self.progress = ctk.CTkProgressBar(
            result_frame,
            width=300,
            height=18
        )

        self.progress.pack(pady=10)

        self.progress.set(0)

    # ==========================
    # Recommendation Box
    # ==========================

        ctk.CTkLabel(
            self.right,
            text="Recommendations",
            font=("Arial",18,"bold")
        ).pack()

        self.recommendation = ctk.CTkTextbox(
            self.right,
            width=380,
            height=170
        )

        self.recommendation.pack(
            pady=10
        )

        self.recommendation.insert(
            "1.0",
            "Patient recommendations will appear here..."
        )

        self.recommendation.configure(
            state="disabled"
        )

    # ======================================================
    # BUTTONS
    # ======================================================

    def create_buttons(self):

        frame = ctk.CTkFrame(
            self.left,
            fg_color="transparent"
        )

        frame.grid(
            row=9,
            column=0,
            columnspan=2,
            pady=35
        )

        self.predict_btn = ctk.CTkButton(
            frame,
            text="🔍 Predict",
            command=self.predict,
            width=140,
            height=40,
            font=("Arial",16,"bold")
        )

        self.predict_btn.grid(
            row=0,
            column=0,
            padx=8
        )

        self.clear_btn = ctk.CTkButton(

            frame,

            text="🧹 Clear",

            command=self.clear_fields,

            width=120,

            height=40,

            font=("Arial",16)

        )

        self.clear_btn.grid(
            row=0,
            column=1,
            padx=8
        )

        self.exit_btn = ctk.CTkButton(

            frame,

            text="❌ Exit",

            width=120,

            height=40,

            fg_color="red",

            hover_color="#8B0000",

            command=self.app.destroy

        )

        self.exit_btn.grid(
            row=0,
            column=2,
            padx=8
        )
    

    def load_model(self):
        import os  # Import os locally or add it to the top of your file

        try:
            # 1. Get the absolute path to the directory where this script lives
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # 2. Construct absolute paths to your files
            scaler_path = os.path.join(script_dir, "scaler.pkl")
            model_path = os.path.join(script_dir, "model.tflite")

            # 3. Load the scaler using the absolute path
            self.scaler = load(scaler_path)

            # 4. Load the TFLite interpreter using the absolute path
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path
            )

            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()

            self.output_details = self.interpreter.get_output_details()

            self.model_status.configure(
                text="TensorFlow Lite Model : Loaded",
                text_color="green"
            )

        except Exception as e:

            self.model_status.configure(
                text="Model Loading Failed",
                text_color="red"
            )

            messagebox.showerror(
                "Error",
                f"Failed to load assets.\n\nLooking in: {script_dir}\n\nError details: {str(e)}"
            )

    def get_patient_data(self):

        gender = 0 if self.gender.get() == "Male" else 1

        hypertension = 1 if self.hypertension.get() == "Yes" else 0

        heart = 1 if self.heart.get() == "Yes" else 0

        smoking_dict = {

            "never":0,

            "No Info":1,

            "current":2,

            "former":3,

            "ever":4,

            "not current":5

        }

        smoking = smoking_dict[self.smoking.get()]

        patient = np.array([[

            gender,

            float(self.age.get()),

            hypertension,

            heart,

            smoking,

            float(self.bmi.get()),

            float(self.hba1c.get()),

            float(self.glucose.get())

        ]])

        patient = self.scaler.transform(patient)

        return patient.astype(np.float32)
    
    def predict(self):

        try:

            patient = self.get_patient_data()

            self.interpreter.set_tensor(

                self.input_details[0]["index"],

                patient

            )

            self.interpreter.invoke()

            prediction = self.interpreter.get_tensor(

                self.output_details[0]["index"]

            )

            score = float(prediction[0][0])

            probability = score * 100

            self.update_dashboard(
                score,
                probability
            )

        except Exception as e:

            messagebox.showerror(
                "Prediction Error",
                str(e)
            )

    def update_dashboard(
        self,
        score,
        probability
    ):

        self.progress.set(score)

        self.risk_score.configure(

            text=f"Risk Score : {score:.4f}"

        )

        self.probability.configure(

            text=f"Probability : {probability:.2f}%"

        )

        if probability < 30:

            level = "LOW"

            color = "green"

            prediction = "No Diabetes Risk"

            recommendation = (

                "✓ Maintain a healthy diet.\n\n"

                "✓ Exercise regularly.\n\n"

                "✓ Annual health check-up."

            )

        elif probability < 70:

            level = "MODERATE"

            color = "orange"

            prediction = "Moderate Diabetes Risk"

            recommendation = (

                "✓ Reduce sugar intake.\n\n"

                "✓ Increase physical activity.\n\n"

                "✓ Monitor blood glucose."

            )

        else:

            level = "HIGH"

            color = "red"

            prediction = "Diabetes Risk Detected"

            recommendation = (

                "✓ Consult a physician immediately.\n\n"

                "✓ Regular blood sugar monitoring.\n\n"

                "✓ Follow a diabetic diet.\n\n"

                "✓ Daily exercise."

            )

        self.result_status.configure(

            text="Prediction Complete"

        )

        self.risk_level.configure(

            text=level,

            text_color=color

        )

        self.prediction.configure(

            text=prediction

        )

        self.recommendation.configure(
            state="normal"
        )

        self.recommendation.delete(
            "1.0",
            "end"
        )

        self.recommendation.insert(
            "1.0",
            recommendation
        )

        self.recommendation.configure(
            state="disabled"
        )

        # ======================================================
    # CLEAR FUNCTION
    # ======================================================

    def clear_fields(self):

        # Clear text entries
        self.age.delete(0, "end")
        self.bmi.delete(0, "end")
        self.hba1c.delete(0, "end")
        self.glucose.delete(0, "end")


        # Reset dropdowns
        self.gender.set("")
        self.hypertension.set("")
        self.heart.set("")
        self.smoking.set("")


        # Reset dashboard

        self.result_status.configure(
            text="Waiting for Prediction"
        )

        self.risk_score.configure(
            text="Risk Score : --"
        )

        self.probability.configure(
            text="Probability : -- %"
        )

        self.risk_level.configure(
            text="Risk Level : --",
            text_color="white"
        )

        self.prediction.configure(
            text="Prediction : --"
        )


        # Reset progress bar

        self.progress.set(0)


        # Reset recommendation box

        self.recommendation.configure(
            state="normal"
        )

        self.recommendation.delete(
            "1.0",
            "end"
        )

        self.recommendation.insert(
            "1.0",
            "Patient recommendations will appear here..."
        )

        self.recommendation.configure(
            state="disabled"
        )

    # ======================================================
    # RUN
    # ======================================================

    def run(self):

        self.app.mainloop()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    app = DiabetesApp()

    app.run()