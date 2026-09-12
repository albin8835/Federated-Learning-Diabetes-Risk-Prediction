import os
import sys
from datetime import datetime
import customtkinter as ctk
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from tkinter import messagebox

# ==========================================================
# THEME & STYLING CONFIGURATION
# ==========================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Modern Medical SaaS Palette
COLOR_BG = "#0B1120"           # Slate 950
COLOR_CARD = "#1E293B"         # Slate 800
COLOR_CARD_BORDER = "#334155"  # Slate 700
COLOR_INPUT_BG = "#0F172A"     # Slate 900
COLOR_TEXT_PRIMARY = "#F8FAFC" # Slate 50
COLOR_TEXT_MUTED = "#94A3B8"   # Slate 400
COLOR_ACCENT = "#0284C7"       # Medical Blue
COLOR_ACCENT_HOVER = "#0369A1"
COLOR_SUCCESS = "#10B981"      # Emerald Green
COLOR_WARNING = "#F59E0B"      # Amber
COLOR_DANGER = "#EF4444"       # Rose Red

FEATURE_NAMES = [
    "Gender", "Age", "Hypertension", "Heart Disease",
    "Smoking History", "BMI", "HbA1c Level", "Blood Glucose"
]

class ModernDiabetesApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("FL-DIAPRED • Federated Learning Clinical Diabetes Assessment")
        self.app.geometry("1280x760")
        self.app.minsize(1180, 720)
        self.app.configure(fg_color=COLOR_BG)

        # State Variables
        self.last_score = None
        self.last_probability = None
        self.last_level = None
        self.last_prediction = None
        self.last_recommendation = None
        self.last_patient_dict = None
        self.last_scaled_patient = None
        self.last_sorted_xai = []

        # Build UI Layout
        self.build_header()
        self.build_main_container()

        # Load AI Engine
        self.load_model()

    # ======================================================
    # HEADER BAR
    # ======================================================
    def build_header(self):
        header = ctk.CTkFrame(
            self.app,
            fg_color=COLOR_CARD,
            corner_radius=0,
            height=70,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        header.pack(fill="x", side="top")

        # Left: Branding & Subtitle
        brand_frame = ctk.CTkFrame(header, fg_color="transparent")
        brand_frame.pack(side="left", padx=25, pady=12)

        title = ctk.CTkLabel(
            brand_frame,
            text="🛡️ FL-DIAPRED",
            font=("Segoe UI", 22, "bold"),
            text_color="#38BDF8"
        )
        title.pack(side="left", padx=(0, 15))

        divider = ctk.CTkLabel(
            brand_frame,
            text="|",
            font=("Segoe UI", 18),
            text_color=COLOR_CARD_BORDER
        )
        divider.pack(side="left", padx=(0, 15))

        subtitle = ctk.CTkLabel(
            brand_frame,
            text="Privacy-Preserving Federated Learning • Clinical Diabetes Screening System",
            font=("Segoe UI", 13),
            text_color=COLOR_TEXT_MUTED
        )
        subtitle.pack(side="left")

        # Right: Hardware/Engine Status Pill
        status_frame = ctk.CTkFrame(
            header,
            fg_color="#0F172A",
            corner_radius=20,
            border_width=1,
            border_color="#1E3A8A"
        )
        status_frame.pack(side="right", padx=25, pady=15)

        self.status_pill = ctk.CTkLabel(
            status_frame,
            text="● Initializing Engine...",
            font=("Segoe UI", 12, "bold"),
            text_color=COLOR_WARNING,
            padx=14,
            pady=4
        )
        self.status_pill.pack()

    # ======================================================
    # MAIN CONTENT (SPLIT VIEW)
    # ======================================================
    def build_main_container(self):
        main = ctk.CTkFrame(self.app, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=24, pady=18)
        main.columnconfigure(0, weight=5)
        main.columnconfigure(1, weight=6)
        main.rowconfigure(0, weight=1)

        self.build_left_patient_card(main)
        self.build_right_dashboard_card(main)

    # ======================================================
    # LEFT CARD: PATIENT CLINICAL FORM
    # ======================================================
    def build_left_patient_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_CARD,
            corner_radius=16,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        # Title
        title_box = ctk.CTkFrame(card, fg_color="transparent")
        title_box.pack(fill="x", padx=22, pady=(18, 10))

        ctk.CTkLabel(
            title_box,
            text="📋 Patient Clinical Profile",
            font=("Segoe UI", 18, "bold"),
            text_color=COLOR_TEXT_PRIMARY
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Enter baseline vitals and clinical measurements for risk evaluation",
            font=("Segoe UI", 12),
            text_color=COLOR_TEXT_MUTED
        ).pack(anchor="w")

        # Two-Column Input Grid
        grid_frame = ctk.CTkFrame(card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=22, pady=6)
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        def make_field(parent, label_text, row, col, is_combo=False, combo_values=None, placeholder=""):
            box = ctk.CTkFrame(parent, fg_color="transparent")
            box.grid(row=row, column=col, padx=8, pady=6, sticky="ew")

            lbl = ctk.CTkLabel(
                box,
                text=label_text,
                font=("Segoe UI", 12, "bold"),
                text_color=COLOR_TEXT_MUTED
            )
            lbl.pack(anchor="w", pady=(0, 3))

            if is_combo:
                widget = ctk.CTkComboBox(
                    box,
                    values=combo_values or [],
                    fg_color=COLOR_INPUT_BG,
                    border_color=COLOR_CARD_BORDER,
                    button_color=COLOR_ACCENT,
                    dropdown_fg_color=COLOR_CARD,
                    font=("Segoe UI", 13),
                    height=36,
                    corner_radius=8
                )
            else:
                widget = ctk.CTkEntry(
                    box,
                    placeholder_text=placeholder,
                    placeholder_text_color="#475569",
                    fg_color=COLOR_INPUT_BG,
                    border_color=COLOR_CARD_BORDER,
                    text_color=COLOR_TEXT_PRIMARY,
                    font=("Segoe UI", 13),
                    height=36,
                    corner_radius=8
                )
            widget.pack(fill="x")
            return widget

        # 1. Gender & Age
        self.gender = make_field(grid_frame, "Gender", 0, 0, True, ["Female", "Male", "Other"])
        self.age = make_field(grid_frame, "Age (years)", 0, 1, placeholder="e.g. 52")

        # 2. Hypertension & Heart Disease
        self.hypertension = make_field(grid_frame, "Hypertension History", 1, 0, True, ["No", "Yes"])
        self.heart = make_field(grid_frame, "Heart Disease History", 1, 1, True, ["No", "Yes"])

        # 3. Smoking History & BMI
        self.smoking = make_field(grid_frame, "Smoking History", 2, 0, True, ["never", "No Info", "current", "former", "ever", "not current"])
        self.bmi = make_field(grid_frame, "BMI (kg/m²)", 2, 1, placeholder="e.g. 28.5")

        # 4. HbA1c & Blood Glucose
        self.hba1c = make_field(grid_frame, "HbA1c Level (%)", 3, 0, placeholder="e.g. 6.8")
        self.glucose = make_field(grid_frame, "Blood Glucose (mg/dL)", 3, 1, placeholder="e.g. 155")

        # Quick Demo Buttons
        demo_frame = ctk.CTkFrame(card, fg_color="transparent")
        demo_frame.pack(fill="x", padx=22, pady=(12, 6))

        ctk.CTkLabel(
            demo_frame,
            text="⚡ Quick Demo Presets:",
            font=("Segoe UI", 11, "bold"),
            text_color="#64748B"
        ).pack(side="left", padx=(0, 10))

        healthy_btn = ctk.CTkButton(
            demo_frame,
            text="🟢 Healthy Baseline",
            width=140,
            height=28,
            font=("Segoe UI", 11, "bold"),
            fg_color="#1E3A5F",
            hover_color="#2A4D78",
            corner_radius=6,
            command=self.load_healthy_sample
        )
        healthy_btn.pack(side="left", padx=4)

        risk_btn = ctk.CTkButton(
            demo_frame,
            text="🔴 High Risk Patient",
            width=140,
            height=28,
            font=("Segoe UI", 11, "bold"),
            fg_color="#5C2424",
            hover_color="#7A3333",
            corner_radius=6,
            command=self.load_high_risk_sample
        )
        risk_btn.pack(side="left", padx=4)

        # Action Buttons Section
        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.pack(fill="x", padx=22, pady=(16, 18), side="bottom")

        self.predict_btn = ctk.CTkButton(
            btn_frame,
            text="🔍 ANALYZE DIABETES RISK",
            command=self.predict,
            height=44,
            font=("Segoe UI", 15, "bold"),
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            corner_radius=10
        )
        self.predict_btn.pack(fill="x", pady=(0, 8))

        sub_btn_frame = ctk.CTkFrame(btn_frame, fg_color="transparent")
        sub_btn_frame.pack(fill="x")
        sub_btn_frame.columnconfigure(0, weight=3)
        sub_btn_frame.columnconfigure(1, weight=1)

        self.export_btn = ctk.CTkButton(
            sub_btn_frame,
            text="📄 Export Clinical Report (PDF)",
            command=self.export_report,
            height=38,
            font=("Segoe UI", 13, "bold"),
            fg_color="#059669",
            hover_color="#047857",
            corner_radius=8
        )
        self.export_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.clear_btn = ctk.CTkButton(
            sub_btn_frame,
            text="🧹 Reset",
            command=self.clear_fields,
            height=38,
            font=("Segoe UI", 13),
            fg_color="#475569",
            hover_color="#334155",
            corner_radius=8
        )
        self.clear_btn.grid(row=0, column=1, sticky="ew")

    # ======================================================
    # RIGHT CARD: DIAGNOSTIC INTELLIGENCE DASHBOARD
    # ======================================================
    def build_right_dashboard_card(self, parent):
        card = ctk.CTkFrame(
            parent,
            fg_color=COLOR_CARD,
            corner_radius=16,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        # Title
        title_box = ctk.CTkFrame(card, fg_color="transparent")
        title_box.pack(fill="x", padx=22, pady=(18, 12))

        ctk.CTkLabel(
            title_box,
            text="📊 Diagnostic Intelligence Dashboard",
            font=("Segoe UI", 18, "bold"),
            text_color=COLOR_TEXT_PRIMARY
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_box,
            text="Real-time neural risk classification & Explainable AI attribution",
            font=("Segoe UI", 12),
            text_color=COLOR_TEXT_MUTED
        ).pack(anchor="w")

        # 1. Primary Risk Card
        self.risk_card = ctk.CTkFrame(
            card,
            fg_color=COLOR_INPUT_BG,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        self.risk_card.pack(fill="x", padx=22, pady=(0, 10))

        risk_top = ctk.CTkFrame(self.risk_card, fg_color="transparent")
        risk_top.pack(fill="x", padx=16, pady=(12, 6))

        self.score_label = ctk.CTkLabel(
            risk_top,
            text="--%",
            font=("Segoe UI", 36, "bold"),
            text_color=COLOR_TEXT_MUTED
        )
        self.score_label.pack(side="left")

        self.level_badge = ctk.CTkLabel(
            risk_top,
            text="READY FOR ANALYSIS",
            font=("Segoe UI", 12, "bold"),
            fg_color="#334155",
            text_color=COLOR_TEXT_PRIMARY,
            corner_radius=12,
            padx=14,
            pady=4
        )
        self.level_badge.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(
            self.risk_card,
            height=14,
            corner_radius=7,
            fg_color="#1E293B",
            progress_color=COLOR_ACCENT
        )
        self.progress_bar.pack(fill="x", padx=16, pady=(4, 8))
        self.progress_bar.set(0)

        self.class_label = ctk.CTkLabel(
            self.risk_card,
            text="Awaiting patient vitals to generate risk assessment.",
            font=("Segoe UI", 12),
            text_color=COLOR_TEXT_MUTED
        )
        self.class_label.pack(anchor="w", padx=16, pady=(0, 12))

        # 2. Explainable AI (XAI) Insight Panel
        xai_frame = ctk.CTkFrame(
            card,
            fg_color=COLOR_INPUT_BG,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        xai_frame.pack(fill="x", padx=22, pady=8)

        ctk.CTkLabel(
            xai_frame,
            text="💡 Explainable AI (XAI) • Top Contributing Factors",
            font=("Segoe UI", 13, "bold"),
            text_color="#38BDF8"
        ).pack(anchor="w", padx=16, pady=(10, 6))

        self.xai_container = ctk.CTkFrame(xai_frame, fg_color="transparent")
        self.xai_container.pack(fill="x", padx=16, pady=(0, 10))

        self.xai_placeholder = ctk.CTkLabel(
            self.xai_container,
            text="Top clinical risk drivers will be highlighted here upon prediction.",
            font=("Segoe UI", 11),
            text_color="#64748B"
        )
        self.xai_placeholder.pack(anchor="w", pady=4)

        # 3. Clinical Recommendations Panel
        rec_frame = ctk.CTkFrame(
            card,
            fg_color=COLOR_INPUT_BG,
            corner_radius=12,
            border_width=1,
            border_color=COLOR_CARD_BORDER
        )
        rec_frame.pack(fill="both", expand=True, padx=22, pady=(8, 16))

        ctk.CTkLabel(
            rec_frame,
            text="🩺 Clinical Lifestyle & Action Plan",
            font=("Segoe UI", 13, "bold"),
            text_color="#34D399"
        ).pack(anchor="w", padx=16, pady=(10, 4))

        self.rec_text = ctk.CTkTextbox(
            rec_frame,
            fg_color="transparent",
            text_color=COLOR_TEXT_PRIMARY,
            font=("Segoe UI", 12),
            wrap="word",
            activate_scrollbars=True
        )
        self.rec_text.pack(fill="both", expand=True, padx=12, pady=(0, 10))
        self.rec_text.insert("1.0", "Patient recommendations and clinical precautions will appear here once vitals are evaluated.")
        self.rec_text.configure(state="disabled")

    # ======================================================
    # QUICK LOAD PRESETS
    # ======================================================
    def load_healthy_sample(self):
        self.gender.set("Female")
        self.age.delete(0, "end"); self.age.insert(0, "28")
        self.hypertension.set("No")
        self.heart.set("No")
        self.smoking.set("never")
        self.bmi.delete(0, "end"); self.bmi.insert(0, "21.4")
        self.hba1c.delete(0, "end"); self.hba1c.insert(0, "4.8")
        self.glucose.delete(0, "end"); self.glucose.insert(0, "88")
        self.predict()

    def load_high_risk_sample(self):
        self.gender.set("Male")
        self.age.delete(0, "end"); self.age.insert(0, "58")
        self.hypertension.set("Yes")
        self.heart.set("Yes")
        self.smoking.set("former")
        self.bmi.delete(0, "end"); self.bmi.insert(0, "34.2")
        self.hba1c.delete(0, "end"); self.hba1c.insert(0, "7.8")
        self.glucose.delete(0, "end"); self.glucose.insert(0, "195")
        self.predict()

    # ======================================================
    # LOAD TFLITE MODEL & SCALER
    # ======================================================
    def load_model(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            scaler_path = os.path.join(script_dir, "scaler.pkl")
            model_path = os.path.join(script_dir, "model.tflite")

            self.scaler = load(scaler_path)
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            self.status_pill.configure(
                text="● TFLite Model Active (Raspberry Pi 4)",
                text_color=COLOR_SUCCESS,
                fg_color="#064E3B"
            )
        except Exception as e:
            self.status_pill.configure(
                text="● Model Load Error",
                text_color=COLOR_DANGER,
                fg_color="#450A0A"
            )
            messagebox.showerror("Model Load Failure", f"Could not load TFLite model or scaler:\n{e}")

    # ======================================================
    # PREDICTION & XAI PIPELINE
    # ======================================================
    def get_patient_data(self):
        gender_val = 0 if self.gender.get() == "Male" else (1 if self.gender.get() == "Female" else 2)
        hypertension_val = 1 if self.hypertension.get() == "Yes" else 0
        heart_val = 1 if self.heart.get() == "Yes" else 0
        smoking_dict = {"never": 0, "No Info": 1, "current": 2, "former": 3, "ever": 4, "not current": 5}
        smoking_val = smoking_dict.get(self.smoking.get(), 1)

        age_val = float(self.age.get().strip())
        bmi_val = float(self.bmi.get().strip())
        hba1c_val = float(self.hba1c.get().strip())
        glucose_val = float(self.glucose.get().strip())

        patient_raw = np.array([[
            gender_val, age_val, hypertension_val, heart_val,
            smoking_val, bmi_val, hba1c_val, glucose_val
        ]], dtype=float)

        self.last_patient_dict = {
            "Gender": self.gender.get(),
            "Age": age_val,
            "Hypertension": self.hypertension.get(),
            "Heart Disease": self.heart.get(),
            "Smoking History": self.smoking.get(),
            "BMI": bmi_val,
            "HbA1c Level": hba1c_val,
            "Blood Glucose": glucose_val
        }

        patient_scaled = self.scaler.transform(patient_raw).astype(np.float32)
        self.last_scaled_patient = patient_scaled
        return patient_scaled

    def predict(self):
        try:
            patient_scaled = self.get_patient_data()

            self.interpreter.set_tensor(self.input_details[0]["index"], patient_scaled)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]["index"])

            score = float(prediction[0][0])
            probability = score * 100

            self.update_dashboard(score, probability)
            self.update_xai(patient_scaled, score)

        except ValueError:
            messagebox.showwarning("Input Validation Error", "Please ensure all numeric fields (Age, BMI, HbA1c, Glucose) are filled with valid numbers.")
        except Exception as e:
            messagebox.showerror("Inference Error", f"Prediction execution failed:\n{e}")

    # ======================================================
    # UPDATE DASHBOARD & VISUAL RISK METRICS
    # ======================================================
    def update_dashboard(self, score, probability):
        self.last_score = score
        self.last_probability = probability

        self.score_label.configure(text=f"{probability:.1f}%")
        self.progress_bar.set(score)

        if probability < 30.0:
            level = "LOW RISK"
            badge_color = COLOR_SUCCESS
            bar_color = COLOR_SUCCESS
            badge_bg = "#064E3B"
            classification = "Normal Health Range • Low Probability of Diabetes"
            recommendations = (
                "✓ Maintain current balanced nutrition and hydration.\n\n"
                "✓ Continue regular aerobic exercise (150 mins/week).\n\n"
                "✓ Routine annual metabolic screening recommended."
            )
        elif probability < 70.0:
            level = "MODERATE RISK"
            badge_color = COLOR_WARNING
            bar_color = COLOR_WARNING
            badge_bg = "#78350F"
            classification = "Pre-Diabetic Indicator • Elevated Glycemic Markers"
            recommendations = (
                "⚠ Reduce refined sugars and high-glycemic carbohydrates.\n\n"
                "⚠ Increase dietary fiber intake and moderate cardio daily.\n\n"
                "⚠ Schedule follow-up Fasting Blood Glucose check within 90 days."
            )
        else:
            level = "HIGH RISK"
            badge_color = COLOR_DANGER
            bar_color = COLOR_DANGER
            badge_bg = "#7F1D1D"
            classification = "Clinical Risk Detected • Immediate Medical Follow-Up Advised"
            recommendations = (
                "🚨 Consult an endocrinologist or primary physician promptly.\n\n"
                "🚨 Daily glycemic monitoring (Fasting & Post-Prandial).\n\n"
                "🚨 Comprehensive diabetic diet plan and clinical evaluation."
            )

        self.last_level = level
        self.last_prediction = classification
        self.last_recommendation = recommendations

        self.level_badge.configure(
            text=f"● {level}",
            text_color=badge_color,
            fg_color=badge_bg
        )
        self.score_label.configure(text_color=badge_color)
        self.progress_bar.configure(progress_color=bar_color)
        self.class_label.configure(
            text=f"Status: {classification} (Risk Score: {score:.4f})"
        )

        self.rec_text.configure(state="normal")
        self.rec_text.delete("1.0", "end")
        self.rec_text.insert("1.0", recommendations)
        self.rec_text.configure(state="disabled")

    # ======================================================
    # UPDATE EXPLAINABLE AI (XAI)
    # ======================================================
    def update_xai(self, patient_scaled, base_prob):
        zero_baseline = np.zeros_like(patient_scaled)
        contributions = {}

        for i, name in enumerate(FEATURE_NAMES):
            perturbed = patient_scaled.copy()
            perturbed[0, i] = zero_baseline[0, i]
            self.interpreter.set_tensor(self.input_details[0]["index"], perturbed)
            self.interpreter.invoke()
            p_pred = self.interpreter.get_tensor(self.output_details[0]["index"])
            impact = base_prob - float(p_pred[0][0])
            contributions[name] = impact

        # Restore original tensor
        self.interpreter.set_tensor(self.input_details[0]["index"], patient_scaled)
        self.interpreter.invoke()

        total_abs = sum(abs(v) for v in contributions.values()) or 1e-6
        percentages = {k: (abs(v) / total_abs) * 100 for k, v in contributions.items()}
        sorted_items = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        self.last_sorted_xai = [(k, v, percentages[k]) for k, v in sorted_items]

        # Clear existing XAI widgets
        for widget in self.xai_container.winfo_children():
            widget.destroy()

        # Render Top 3 Risk Factors
        for feat, imp, pct in self.last_sorted_xai[:3]:
            row = ctk.CTkFrame(self.xai_container, fg_color="transparent")
            row.pack(fill="x", pady=2)

            direction = "▲ Increases Risk" if imp >= 0 else "▼ Lowers Risk"
            tag_color = COLOR_DANGER if imp >= 0 else COLOR_SUCCESS

            name_lbl = ctk.CTkLabel(
                row,
                text=f"{feat} ({self.last_patient_dict.get(feat, '')})",
                font=("Segoe UI", 11, "bold"),
                text_color=COLOR_TEXT_PRIMARY,
                width=160,
                anchor="w"
            )
            name_lbl.pack(side="left")

            dir_lbl = ctk.CTkLabel(
                row,
                text=f"{direction} ({pct:.1f}%)",
                font=("Segoe UI", 11),
                text_color=tag_color,
                anchor="w"
            )
            dir_lbl.pack(side="left", padx=8)

            bar = ctk.CTkProgressBar(
                row,
                height=6,
                corner_radius=3,
                fg_color="#334155",
                progress_color=tag_color
            )
            bar.pack(side="right", fill="x", expand=True, padx=4)
            bar.set(min(pct / 100.0, 1.0))

    # ======================================================
    # RESET FORM
    # ======================================================
    def clear_fields(self):
        for entry in [self.age, self.bmi, self.hba1c, self.glucose]:
            entry.delete(0, "end")
        self.gender.set("Female")
        self.hypertension.set("No")
        self.heart.set("No")
        self.smoking.set("never")

        self.score_label.configure(text="--%", text_color=COLOR_TEXT_MUTED)
        self.progress_bar.set(0)
        self.progress_bar.configure(progress_color=COLOR_ACCENT)
        self.level_badge.configure(text="READY FOR ANALYSIS", text_color=COLOR_TEXT_PRIMARY, fg_color="#334155")
        self.class_label.configure(text="Awaiting patient vitals to generate risk assessment.")

        for widget in self.xai_container.winfo_children():
            widget.destroy()
        self.xai_placeholder = ctk.CTkLabel(
            self.xai_container,
            text="Top clinical risk drivers will be highlighted here upon prediction.",
            font=("Segoe UI", 11),
            text_color="#64748B"
        )
        self.xai_placeholder.pack(anchor="w", pady=4)

        self.rec_text.configure(state="normal")
        self.rec_text.delete("1.0", "end")
        self.rec_text.insert("1.0", "Patient recommendations and clinical precautions will appear here once vitals are evaluated.")
        self.rec_text.configure(state="disabled")

        self.last_score = None
        self.last_probability = None
        self.last_level = None
        self.last_patient_dict = None
        self.last_scaled_patient = None
        self.last_sorted_xai = []

    # ======================================================
    # CLINICAL PDF REPORT EXPORT
    # ======================================================
    def export_report(self):
        if self.last_score is None:
            messagebox.showwarning("No Assessment", "Please fill in patient vitals and click 'ANALYZE DIABETES RISK' before exporting a report.")
            return

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            )
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

            script_dir = os.path.dirname(os.path.abspath(__file__))
            reports_dir = os.path.join(script_dir, "generated_reports")
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = os.path.join(reports_dir, f"Clinical_Report_{timestamp}.pdf")

            doc = SimpleDocTemplate(
                pdf_filename,
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=30,
                bottomMargin=30
            )
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "TitleStyle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=colors.HexColor("#0F172A"),
                alignment=1
            )
            sub_style = ParagraphStyle(
                "SubStyle",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=13,
                textColor=colors.HexColor("#475569"),
                alignment=1
            )
            h2_style = ParagraphStyle(
                "H2Style",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=12,
                leading=15,
                textColor=colors.HexColor("#0284C7"),
                spaceBefore=10,
                spaceAfter=4
            )
            body_style = ParagraphStyle(
                "BodyStyle",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#1E293B")
            )
            bold_style = ParagraphStyle(
                "BoldStyle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#0F172A")
            )

            elements = []
            elements.append(Paragraph("🩺 DIABETES RISK ASSESSMENT CLINICAL REPORT", title_style))
            elements.append(Spacer(1, 3))
            elements.append(Paragraph("Federated Learning Framework &bull; Privacy-Preserving Neural Network &bull; Edge Deployment (Raspberry Pi 4)", sub_style))
            elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceBefore=6, spaceAfter=10))

            now_formatted = datetime.now().strftime("%B %d, %Y - %I:%M %p")
            meta_table = Table([
                [Paragraph(f"<b>Assessment Date:</b> {now_formatted}", body_style), Paragraph(f"<b>Report ID:</b> DBR-{timestamp[-6:]}", body_style)],
                [Paragraph("<b>Platform:</b> Raspberry Pi 4 Model B (Edge AI)", body_style), Paragraph("<b>Inference Engine:</b> TensorFlow Lite FedAvg", body_style)]
            ], colWidths=[270, 270])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0,0), (-1,-1), 5)
            ]))
            elements.append(meta_table)
            elements.append(Spacer(1, 8))

            # Risk Summary Card
            is_low = "LOW" in self.last_level
            is_mod = "MODERATE" in self.last_level
            level_color = "#065F46" if is_low else ("#92400E" if is_mod else "#991B1B")
            level_bg = "#D1FAE5" if is_low else ("#FEF3C7" if is_mod else "#FEE2E2")

            risk_card = Table([
                [Paragraph("<b>CLINICAL RISK CLASSIFICATION</b>", bold_style), Paragraph("<b>ASSESSMENT STATUS</b>", bold_style)],
                [
                    Paragraph(f"<font size=14 color='{level_color}'><b>{self.last_level}</b></font><br/>Probability: <b>{self.last_probability:.2f}%</b> | Score: <b>{self.last_score:.4f}</b>", body_style),
                    Paragraph(f"<font size=11 color='{level_color}'><b>{self.last_prediction}</b></font>", body_style)
                ]
            ], colWidths=[270, 270])
            risk_card.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
                ('BACKGROUND', (0,1), (-1,1), colors.HexColor(level_bg)),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor(level_color)),
                ('PADDING', (0,0), (-1,-1), 8)
            ]))
            elements.append(risk_card)
            elements.append(Spacer(1, 10))

            # Patient Vitals Table
            elements.append(Paragraph("1. Patient Profile & Clinical Vitals", h2_style))
            p = self.last_patient_dict
            vitals_data = [
                [Paragraph("<b>Parameter</b>", bold_style), Paragraph("<b>Value</b>", bold_style), Paragraph("<b>Parameter</b>", bold_style), Paragraph("<b>Value</b>", bold_style)],
                [Paragraph("Gender", body_style), Paragraph(str(p["Gender"]), body_style), Paragraph("Smoking History", body_style), Paragraph(str(p["Smoking History"]), body_style)],
                [Paragraph("Age", body_style), Paragraph(f"{p['Age']:.0f} yrs", body_style), Paragraph("Body Mass Index (BMI)", body_style), Paragraph(f"{p['BMI']:.1f} kg/m²", body_style)],
                [Paragraph("Hypertension", body_style), Paragraph(str(p["Hypertension"]), body_style), Paragraph("HbA1c Level", body_style), Paragraph(f"{p['HbA1c Level']:.1f}%", body_style)],
                [Paragraph("Heart Disease", body_style), Paragraph(str(p["Heart Disease"]), body_style), Paragraph("Blood Glucose Level", body_style), Paragraph(f"{p['Blood Glucose']:.0f} mg/dL", body_style)]
            ]
            t_vitals = Table(vitals_data, colWidths=[135, 135, 135, 135])
            t_vitals.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0,0), (-1,-1), 4)
            ]))
            elements.append(t_vitals)
            elements.append(Spacer(1, 10))

            # Explainable AI Table
            elements.append(Paragraph("2. Explainable AI (XAI) - Key Risk Drivers", h2_style))
            xai_data = [[
                Paragraph("<b>Clinical Factor</b>", bold_style),
                Paragraph("<b>Impact Direction</b>", bold_style),
                Paragraph("<b>Relative Contribution</b>", bold_style)
            ]]
            for feat, imp, pct in self.last_sorted_xai:
                dir_text = "<font color='#DC2626'><b>▲ Increases Risk</b></font>" if imp >= 0 else "<font color='#16A34A'><b>▼ Decreases Risk</b></font>"
                xai_data.append([
                    Paragraph(feat, body_style),
                    Paragraph(dir_text, body_style),
                    Paragraph(f"<b>{pct:.1f}%</b>", bold_style)
                ])
            t_xai = Table(xai_data, colWidths=[200, 180, 160])
            t_xai.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0,0), (-1,-1), 4)
            ]))
            elements.append(t_xai)
            elements.append(Spacer(1, 10))

            # Recommendations
            elements.append(Paragraph("3. Clinical Lifestyle & Follow-Up Guidance", h2_style))
            rec_clean = self.last_recommendation.replace('\n\n', '<br/>').replace('\n', ' ').replace('✓', '&bull;').replace('⚠', '&bull;').replace('🚨', '&bull;')
            rec_table = Table([[Paragraph(f"<b>Action Plan:</b><br/>{rec_clean}", body_style)]], colWidths=[540])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
                ('PADDING', (0,0), (-1,-1), 8)
            ]))
            elements.append(rec_table)
            elements.append(Spacer(1, 10))

            # Disclaimer
            elements.append(Paragraph(
                "<b>Medical Disclaimer:</b> Generated by a research Federated Learning neural network for screening purposes only. Not intended as a substitute for professional medical diagnosis.",
                sub_style
            ))

            doc.build(elements)
            messagebox.showinfo("Report Exported", f"Clinical PDF Report successfully generated!\n\nFile saved to:\n{pdf_filename}")

        except Exception as ex:
            messagebox.showerror("Export Error", f"Could not generate PDF report:\n{ex}")

    def run(self):
        self.app.mainloop()

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    app = ModernDiabetesApp()
    app.run()
