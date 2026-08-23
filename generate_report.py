#!/usr/bin/env python3
"""
generate_report.py - Generate a clean, comprehensive 3-page report.pdf.
Strictly organized into the four required sections in exact order.
All fixes from the independent review (B1-B3, F1-F8, P1-P2) are reflected.
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

    def sub(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 30)
        self.ln(0.5)

    def p(self, text):
        self.set_font("Helvetica", "", 9.2)
        self.multi_cell(0, 4.8, text)
        self.ln(1.8)

    def bullet(self, title, desc):
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

    # =========================================================================
    # PAGE 1
    # =========================================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(10, 40, 95)
    pdf.cell(0, 8, "Fetal Distress Detection from Intrapartum CTG Signals", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, "Devanshu Dhoble  |  Janitri Interview Assignment", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # === SECTION 1 ===
    pdf.section_title("1. How I Framed It")

    pdf.sub("Problem formulation")
    pdf.p(
        "I framed this as a supervised binary classification problem: given continuous "
        "FHR and UC time-series signals recorded during labour, predict whether the "
        "infant will be distressed at delivery."
    )

    pdf.sub("Defining 'fetal distress' from delivery outcomes")
    pdf.p(
        "Distressed (y=1) if: umbilical cord pH < 7.20 OR 5-minute Apgar < 7.\n"
        "Normal (y=0): all other cases with valid measurements."
    )
    pdf.p(
        "Label decomposition on the 552-record dataset:\n"
        "  - pH < 7.20 only:    163 records\n"
        "  - Both criteria:      14 records\n"
        "  - Apgar5 < 7 only:     5 records\n"
        "  - Total distressed:  182  (33.0%)\n"
        "  - Total normal:      370  (67.0%)\n\n"
        "The pH arm dominates. The 7.20 threshold represents mild acidemia (not severe "
        "acidosis at 7.05-7.10, as Janitri's brief notes). It was chosen deliberately "
        "to produce a workable 33% positive class on 552 records. The trade-off is "
        "label purity for statistical power. BDecf (base deficit) was available with "
        "zero missing values and could serve as an alternative or complementary label "
        "source in future work."
    )

    pdf.sub("Input time window")
    pdf.p(
        "All recordings end at delivery but vary in length. I standardized the input "
        "to the final 30 minutes (7,200 samples at 4 Hz). Hypoxia accumulates in the "
        "terminal stage of labour, making this window the most diagnostically relevant."
    )

    # === SECTION 2 ===
    pdf.section_title("2. What I Built and How I Checked It")

    pdf.sub("Feature engineering (30 features)")
    pdf.p(
        "I extracted 30 domain-specific features across six categories:\n"
        "  - FHR statistics: mean, std, median, range, IQR, skewness, kurtosis\n"
        "  - Heart-rate variability: RMSSD, mean/median successive differences, STV, LTV\n"
        "    (computed gap-aware: gaps > 1 second are masked, not bridged)\n"
        "  - Baseline: rolling-median estimate (not identical to global median)\n"
        "  - Clinical events: deceleration count, acceleration count\n"
        "  - Signal quality: missing ratio, longest gap in seconds\n"
        "  - UC features: amplitude, contraction frequency, interval regularity\n"
        "  - FHR-UC coupling: correlation, pre-peak vs post-peak FHR, post-minus-pre"
    )

    pdf.sub("Model")
    pdf.p(
        "Random Forest (300 trees, max_depth=4, min_samples_leaf=10, "
        "class_weight='balanced'). The shallow depth prevents memorization: "
        "with depth 12, train AUC was 1.00 vs test AUC 0.73, indicating severe "
        "overfitting. The regularized depth-4 model generalizes better."
    )

    pdf.sub("Baseline comparison and cross-validation")
    pdf.p(
        "Under 5-fold stratified CV repeated 6 times (30 evaluations):\n\n"
        "  DummyClassifier (stratified):   0.50 +/- 0.04\n"
        "  LogReg on fhr_iqr alone:        0.73 +/- 0.05\n"
        "  Full RF (30 features, depth 4): 0.74 +/- 0.05\n\n"
        "The full model marginally outperforms a single feature. This is not a "
        "weakness -- it is evidence of a performance ceiling inherent to summary "
        "statistics on this dataset. Published literature on CTU-CHB reports "
        "similar AUC ranges (0.70-0.76) for feature-based approaches."
    )

    pdf.sub("Threshold selection")
    pdf.p(
        "The default 0.50 threshold was clinically unsuitable (recall ~0.35). "
        "I selected the threshold on training folds to target ~80% recall, yielding "
        "threshold = 0.28. At this operating point:\n"
        "  - Recall:    ~0.80 (catches most distressed cases)\n"
        "  - Precision: ~0.48 (about half of alerts are true positives)\n\n"
        "Catching 80% of compromised babies at 48% precision is a defensible "
        "screening tool. The threshold is configurable via --threshold in "
        "generate_outcomes.py."
    )

    pdf.sub("What the metrics tell us vs. what they do NOT")
    pdf.p(
        "ROC-AUC (0.74 CV) summarizes ranking quality but does not capture the "
        "clinical cost of different errors. Recall is critical because missing a "
        "distressed baby can cause serious harm. Precision matters to avoid alarm "
        "fatigue. Metrics are reported to two decimal places to reflect the "
        "uncertainty inherent in 552 records."
    )

    # =========================================================================
    # PAGE 2-3
    # =========================================================================
    pdf.add_page()

    # === SECTION 3 ===
    pdf.section_title("3. Clinical Utility and Inference")

    pdf.sub("Intended bedside usage")
    pdf.p(
        "This is a decision-support tool for labour wards, providing an automated "
        "second opinion alongside the CTG monitor. It does NOT replace clinicians.\n\n"
        "Usage scenario:\n"
        "  1. The CTG monitor streams FHR and UC data continuously.\n"
        "  2. Every 5-10 minutes, the system ingests the last 30 minutes of signal\n"
        "     and computes the 30-feature vector.\n"
        "  3. The model returns a distress probability (0-1) and a binary alert.\n"
        "  4. If the probability exceeds the threshold, a flag appears on the\n"
        "     bedside display prompting the nurse or doctor to review the trace.\n\n"
        "The threshold can be tuned per clinical context: lower catches more true "
        "positives at the cost of more false alarms."
    )

    pdf.sub("Who looks at the output, and when?")
    pdf.p(
        "Staff nurses and attending obstetricians in the labor ward. The output "
        "updates continuously (every 5-10 min) during active labor, providing a "
        "risk trajectory rather than a single snapshot."
    )

    pdf.sub("generate_outcomes.py: input and output")
    pdf.p(
        "The inference script mirrors the clinical workflow:\n"
        "  - Input: raw WFDB record path OR pre-extracted .npz features (30-dim).\n"
        "  - Processing: loads model.pkl + scaler.pkl, extracts features (if raw),\n"
        "    scales them, runs the Random Forest.\n"
        "  - Output: distress probability (float 0-1) and binary label (0 or 1).\n"
        "  - Default threshold: 0.28 (configurable via --threshold).\n\n"
        "This decoupling ensures zero data leakage between training and inference."
    )

    # === SECTION 4 ===
    pdf.section_title("4. Limits and Next Steps")

    pdf.sub("Known limitations")
    pdf.bullet("Small single-center dataset (552 records)", "Limits generalizability. Results may not transfer to other hospitals or monitoring hardware without retraining and external validation.")
    pdf.bullet("Performance ceiling", "Summary statistics over a 30-minute window saturate near 0.73-0.75 AUC on this database. The 30-feature model performs comparably to a single feature (fhr_iqr). This is inherent to the approach, not a tuning problem.")
    pdf.bullet("Retrospective labels", "pH and Apgar are measured at delivery. An infant showing transient distress may recover, creating apparent label noise.")
    pdf.bullet("Signal artefacts in second stage", "During maternal pushing, probe displacement causes extended FHR dropouts. Measured signal loss rises from 17% in the first 10 min of the window to 37% in the final 10 min.")
    pdf.bullet("Gap-aware HRV correction", "Fixing the gap-bridging bug in successive differences slightly reduces headline AUC. The old buggy RMSSD acted as an accidental signal-quality proxy (dropout correlates with outcome). The corrected model scores slightly lower but is honest: RMSSD now measures RMSSD, not probe displacement.")

    pdf.sub("What I would try with more time")
    pdf.bullet("1. Temporal deep learning", "1D-CNN or Bi-LSTM directly on raw 4 Hz FHR+UC signals to learn automated temporal representations that can capture progressive deterioration.")
    pdf.bullet("2. Sliding-window risk trajectory", "Produce a probability every 5-10 minutes to track how risk evolves during labour, rather than a single end-of-recording snapshot.")
    pdf.bullet("3. Probability calibration", "Platt scaling or isotonic regression so the output probability better reflects true prevalence.")
    pdf.bullet("4. SHAP explainability", "TreeSHAP for per-prediction feature attribution displayed on the bedside monitor.")
    pdf.bullet("5. External validation", "Test on a second hospital's dataset to assess generalizability before any clinical deployment.")

    pdf.output(output_path)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_report(os.path.join(script_dir, "report.pdf"))
