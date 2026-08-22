#!/usr/bin/env python3
"""
generate_report.py - Generate a clean, comprehensive 3-page report.pdf for the Janitri assignment.
Strictly organized into the four required sections in exact order.
"""

import os
from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 7, "Janitri Technical Assignment | Fetal Distress Detection from CTG", align="L")
        self.cell(0, 7, "Devanshu Dhoble", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(15, 60, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(20, 20, 20)
        self.ln(1)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 30)
        self.ln(0.5)

    def body_p(self, text):
        self.set_font("Helvetica", "", 9.2)
        self.multi_cell(0, 4.8, text)
        self.ln(1.8)

    def bullet_item(self, title, desc):
        self.set_font("Helvetica", "B", 9.2)
        self.cell(4)
        self.cell(4, 4.8, "-")
        self.cell(self.get_string_width(title) + 1, 4.8, title)
        self.set_font("Helvetica", "", 9.2)
        self.multi_cell(0, 4.8, f": {desc}")
        self.ln(1)


def build_report(output_path="report.pdf"):
    pdf = ReportPDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(12, 12, 12)

    # -------------------------------------------------------------------------
    # PAGE 1
    # -------------------------------------------------------------------------
    pdf.add_page()

    # Document Header Title
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(10, 40, 95)
    pdf.cell(0, 8, "Machine Learning System for Intrapartum Fetal Distress Detection", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Candidate: Devanshu Dhoble  |  Role: AI/ML Engineer  |  Organization: Janitri", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # =========================================================================
    # SECTION 1: How I framed it
    # =========================================================================
    pdf.section_title("1. How I Framed It")

    pdf.subsection_title("Problem Formulation & Clinical Context")
    pdf.body_p(
        "During labour, uterine contractions (UC) periodically compress the placental vessels, temporarily diminishing "
        "maternal-fetal blood and oxygen exchange. A healthy fetus possesses physiological reserve and recovers between "
        "contractions. However, when oxygen delivery is persistently inadequate (due to cord compression, placental "
        "insufficiency, or hyperstimulation), the fetus transitions to anaerobic metabolism, leading to lactic acid "
        "accumulation and progressive metabolic acidosis. Clinicians monitor Cardiotocography (CTG) - tracking Fetal Heart "
        "Rate (FHR) and Uterine Contractions (UC) simultaneously - to identify early physiological signs of hypoxia."
    )
    pdf.body_p(
        "I framed this task as a supervised binary classification problem: given the continuous FHR and UC time-series "
        "signals during active labour, predict whether the infant will suffer from intrapartum distress/hypoxia at delivery."
    )

    pdf.subsection_title("Defining 'Fetal Distress' from Delivery Outcomes")
    pdf.body_p(
        "Delivery outcomes in the CTU-CHB database provide both biochemical measurements (umbilical cord blood gas: pH, "
        "BDecf, BE) and clinical vitality assessments (Apgar scores at 1 and 5 minutes). I defined a composite label:"
    )
    pdf.body_p(
        "    Distressed (y = 1) if:  (Umbilical Cord pH < 7.20)  OR  (5-minute Apgar Score < 7)\n"
        "    Not Distressed / Normal (y = 0):  All other cases with valid measurements."
    )
    pdf.body_p("Rationale for this threshold:")
    pdf.bullet_item("Biochemical Basis (pH < 7.20)", "International obstetrics guidelines (ACOG, FIGO) establish pH < 7.20 as the clinical threshold for mild-to-significant intrapartum acidemia. Severe acidosis is marked at pH < 7.05-7.10, but 7.20 serves as the primary alert threshold where cellular oxygen debt begins.")
    pdf.bullet_item("Clinical Depressed State (Apgar5 < 7)", "The 5-minute Apgar score measures post-delivery physiological recovery (heart rate, respiration, muscle tone, reflexes, skin color). A score below 7 signifies neonate depression requiring resuscitation.")
    pdf.bullet_item("The Composite 'OR' Rule", "Combining biochemical and clinical markers ensures we capture both silent acidotic infants (low pH but compensated Apgar) and clinically depressed neonates. On the 552-record dataset, this produces 182 distressed cases (33.0%) and 370 normal cases (67.0%), providing sufficient positive examples for balanced machine learning.")

    pdf.subsection_title("Input Time Window Strategy")
    pdf.body_p(
        "Labor recordings range from 40 minutes to multiple hours, but all recordings terminate at delivery. Because hypoxia "
        "and metabolic acid accumulate progressively, the terminal stage of labor contains the most discriminative physiological "
        "signals. I standardized the input window to the final 30 minutes before delivery (7,200 time steps at 4 Hz) for all recordings."
    )

    # -------------------------------------------------------------------------
    # PAGE 2
    # -------------------------------------------------------------------------
    pdf.add_page()

    # =========================================================================
    # SECTION 2: What I built and how I checked it
    # =========================================================================
    pdf.section_title("2. What I Built and How I Checked It")

    pdf.subsection_title("Signal Preprocessing & Feature Engineering")
    pdf.body_p(
        "Ultrasound fetal heart rate monitoring suffers from acoustic loss and maternal movement artefacts. I designed a "
        "robust preprocessing pipeline: FHR values <= 50 bpm, >= 250 bpm, or equal to 0 were masked as NaN dropouts. "
        "Rather than relying on black-box deep learning on a modest 552-patient dataset, I engineered 29 domain-specific "
        "clinical features across six physiological categories:"
    )
    pdf.bullet_item("Baseline & Distribution", "FHR mean, standard deviation, median, range, interquartile range (IQR), skewness, kurtosis, and rolling baseline heart rate.")
    pdf.bullet_item("Heart Rate Variability (HRV)", "Root Mean Square of Successive Differences (RMSSD), SDNN, Short-Term Variability (STV proxy over 1-min epochs), and Long-Term Variability (LTV proxy).")
    pdf.bullet_item("Decelerations & Accelerations", "Clinically defined deceleration count (FHR drop >= 15 bpm for >= 15s) and acceleration count relative to estimated baseline.")
    pdf.bullet_item("Signal Reliability", "FHR missing ratio (artefact/dropout fraction) to measure monitoring stability.")
    pdf.bullet_item("Uterine Activity & Regularity", "UC mean, std, max amplitude, contraction frequency per minute, mean contraction interval, and interval standard deviation via peak prominence detection.")
    pdf.bullet_item("FHR-UC Coupling Dynamics", "Pearson cross-correlation between FHR and UC signals, and mean FHR during contractions vs. between contractions (quantifying late deceleration trends).")

    pdf.subsection_title("Model Architecture & Training Setup")
    pdf.body_p(
        "I selected a Random Forest Classifier (300 estimators, max depth 12) combined with a StandardScaler. Key justifications: "
        "(1) Handles non-linear feature interactions without overfitting small sample sizes; (2) Inherently robust to multi-scale features; "
        "(3) Employs 'class_weight=balanced' to penalize misclassification of the distressed minority class; (4) Provides intrinsic feature "
        "importance rankings for medical explainability. The dataset (552 cases) was split 80/20 into 441 training samples and 111 held-out test samples using stratified sampling."
    )

    pdf.subsection_title("Test Set Evaluation & Metric Analysis")
    pdf.body_p(
        "The model was evaluated on the independent 111-sample test set. The exact test performance metrics obtained are:"
    )

    # Metrics Summary Box / Table
    pdf.set_fill_color(242, 246, 252)
    pdf.set_draw_color(180, 200, 230)
    pdf.rect(12, pdf.get_y(), 186, 24, "FD")
    pdf.set_xy(14, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 9.5)
    pdf.cell(46, 5, "ROC-AUC: 0.7330", align="L")
    pdf.cell(46, 5, "Accuracy: 70.27%", align="L")
    pdf.cell(46, 5, "Specificity: 87.84%", align="L")
    pdf.cell(46, 5, "Precision: 59.09%", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(14)
    pdf.cell(46, 5, "Recall (Sens.): 35.14%", align="L")
    pdf.cell(46, 5, "F1-Score: 0.4407", align="L")
    pdf.cell(92, 5, "Confusion Matrix: TP=13, FP=9, FN=24, TN=65", align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.body_p("What the metrics tell us vs. what they do NOT tell us:")
    pdf.bullet_item("ROC-AUC (0.7330)", "Tells us the model possesses solid, generalizable ranking capability across diverse probability thresholds (significantly above 0.50 random guessing). However, it does not reveal the operational performance at a specific clinical alert cut-off.")
    pdf.bullet_item("Specificity (0.8784)", "Tells us that 88% of healthy fetuses are correctly classified without false alerts. This is critical to prevent clinician alarm fatigue and avoid unnecessary emergency Caesarean deliveries.")
    pdf.bullet_item("Precision (0.5909)", "Indicates that nearly 60% of triggered alarms represent true distress cases.")
    pdf.bullet_item("Recall Gap (0.3514)", "Shows that at the default 0.50 decision threshold, the model catches 13 of 37 distressed fetuses, missing 24. In clinical practice, missing a compromised fetus is catastrophic; this demonstrates that a fixed 0.50 threshold is clinically unsuitable and must be calibrated downward to prioritize sensitivity.")

    # -------------------------------------------------------------------------
    # PAGE 3
    # -------------------------------------------------------------------------
    pdf.add_page()

    # =========================================================================
    # SECTION 3: Clinical utility and inference
    # =========================================================================
    pdf.section_title("3. Clinical Utility and Inference")

    pdf.subsection_title("Real-World Bedside Usage Scenario")
    pdf.body_p(
        "This tool is designed as an Intelligent Bedside Decision Support System (CDSS) for labour and delivery wards. It does "
        "not replace obstetricians or midwives; rather, it provides an automated, continuous, objective 'second pair of eyes'."
    )
    pdf.bullet_item("Who monitors the output?", "Staff nurses, resident medical officers, and attending obstetricians in the labor ward.")
    pdf.bullet_item("When does it run?", "Continuously during active labor. Every 5 to 10 minutes, a background sliding window ingests the most recent 30 minutes of streaming CTG data and updates a real-time Risk Trajectory on the bedside monitor.")
    pdf.bullet_item("What does it return?", "(1) Continuous Distress Risk Index (0.0 to 1.0); (2) Visual Risk State (Green = Normal, Amber = Observe, Red = Urgent Review); (3) Top contributing feature factors (e.g. 'Repetitive Decelerations' or 'Depressed STV').")

    pdf.subsection_title("Alignment of generate_outcomes.py with Clinical Reality")
    pdf.body_p(
        "The standalone inference script (generate_outcomes.py) precisely reflects this operational architecture. It decouples "
        "training from inference, accepting either raw WFDB streaming channels or pre-extracted feature vectors, applying the "
        "frozen StandardScaler transformation and Random Forest inference, and returning probability scores alongside configurable "
        "threshold-based alerts. This ensures zero data leakage and seamless integration into edge-device CTG hardware."
    )

    # =========================================================================
    # SECTION 4: Limits and next steps
    # =========================================================================
    pdf.section_title("4. Limits and Next Steps")

    pdf.subsection_title("Honest Appraisal of Weaknesses and Failure Modes")
    pdf.bullet_item("Modest Sample Size (552 cases)", "552 recordings collected from a single university hospital (CTU-CHB) limits physiological diversity and risks population overfitting. Validation across multi-center cohorts is essential.")
    pdf.bullet_item("Static Summary vs. Temporal Evolution", "Aggregating 30 minutes into a static 29-feature vector loses sequential temporal dynamics (e.g., whether decelerations are worsening or recovering over consecutive contractions).")
    pdf.bullet_item("Retrospective Delivery Ground Truth", "Umbilical cord pH and Apgar scores are measured post-delivery. An infant showing transient distress at minute -30 might recover before birth, leading to apparent 'false positive' labels.")
    pdf.bullet_item("Signal Artefacts in Active 2nd Stage", "During maternal pushing, ultrasound probe displacement causes extended dropouts, degrading HRV feature fidelity.")

    pdf.subsection_title("Concrete Engineering Roadmap (What I Would Try with More Time)")
    pdf.bullet_item("1. Dynamic Risk Trajectory & Sliding Windows", "Replace static single-window evaluation with continuous temporal risk tracking over consecutive 10-minute intervals.")
    pdf.bullet_item("2. Temporal Deep Learning Architectures", "Train 1D Convolutional Neural Networks (1D-CNN) paired with Bi-directional LSTMs directly on multi-channel raw FHR+UC signals to learn automated spatial-temporal representations.")
    pdf.bullet_item("3. Clinically Calibrated Thresholds & Platt Scaling", "Tune operational alert thresholds to achieve >= 80% recall (sensitivity), utilizing cost-sensitive learning matrices to penalize false negatives heavily.")
    pdf.bullet_item("4. Explainable AI via SHAP", "Integrate TreeSHAP to display real-time feature attribution directly on the nurse's monitor (e.g., 'Warning: Risk elevated by 32% due to late decelerations').")

    pdf.output(output_path)
    print(f"Report generated successfully at: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_report(os.path.join(script_dir, "report.pdf"))
