#!/usr/bin/env python3
"""
generate_report.py - Generate report.pdf for the Janitri CTG assignment.
Uses fpdf2 to create a clean 2-3 page PDF.
"""

from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "CTG Fetal Distress Detection - Technical Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)


def build_report(output_path="report.pdf"):
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ---- Title ----
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Fetal Distress Detection from CTG Signals", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Devanshu Dhoble", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Janitri Interview Assignment", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ==================================================================
    # Section 1: How I Framed It
    # ==================================================================
    pdf.section_title("1. How I Framed It")

    pdf.body_text(
        "I framed this as a binary classification problem: given the CTG signals recorded "
        "during labour, predict whether the newborn will be distressed at delivery."
    )

    pdf.subsection_title("Label definition")
    pdf.body_text(
        "A recording is labelled distressed (positive class, y=1) when either:\n"
        "  - Umbilical-cord pH < 7.20, or\n"
        "  - 5-minute Apgar score < 7.\n\n"
        "Rationale: pH < 7.20 is a widely-used threshold for clinically significant "
        "metabolic acidosis (ACOG, FIGO guidelines). A 5-min Apgar below 7 indicates "
        "the baby needed resuscitation and may have been compromised. Using the OR "
        "rule captures both biochemical and clinical indicators of distress and produces "
        "a larger positive class (~33% of recordings), which helps the model learn "
        "meaningful patterns from a modest 552-recording dataset.\n\n"
        "Final dataset: 552 recordings total, 370 normal (67%), 182 distressed (33%)."
    )

    pdf.subsection_title("Input representation")
    pdf.body_text(
        "Rather than feeding raw 4 Hz signals directly into a model (which would require "
        "much more data), I extract 29 hand-crafted time-domain features from the last "
        "30 minutes of each recording. These features capture:\n"
        "  - FHR statistics (mean, std, median, range, IQR, skewness, kurtosis)\n"
        "  - Heart-rate variability (RMSSD, SDNN, short/long-term variability)\n"
        "  - Clinical events (count of decelerations and accelerations)\n"
        "  - Uterine contraction patterns (frequency, amplitude, interval regularity)\n"
        "  - FHR-UC coupling (correlation, FHR during vs. between contractions)\n"
        "  - Signal quality (fraction of missing/artefact FHR values)\n\n"
        "These features are motivated by clinical CTG interpretation guidelines (FIGO) "
        "and prior literature on computerised CTG analysis."
    )

    # ==================================================================
    # Section 2: What I Built and How I Checked It
    # ==================================================================
    pdf.section_title("2. What I Built and How I Checked It")

    pdf.subsection_title("Data processing")
    pdf.body_text(
        "I loaded all 552 recordings from the CTU-CHB database using the wfdb library. "
        "For each recording I:\n"
        "  1. Read the physical (scaled) FHR and UC signals from the .dat file.\n"
        "  2. Cleaned the FHR signal by replacing out-of-range values (<50 or >250 bpm) "
        "and zeros with NaN.\n"
        "  3. Extracted the last 30 minutes of data (7200 samples at 4 Hz).\n"
        "  4. Computed the 29-feature vector.\n"
        "  5. Parsed the .hea header to extract pH, BDecf, Apgar1, Apgar5.\n\n"
        "All 552 records were successfully processed. The data was split 80/20 into "
        "train (441 samples) and test (111 samples) sets, stratified by label."
    )

    pdf.subsection_title("Model choice")
    pdf.body_text(
        "I chose a Random Forest classifier (scikit-learn) with 300 trees and max "
        "depth 12. Key reasons:\n"
        "  - Works well on small tabular datasets without extensive hyperparameter tuning.\n"
        "  - Handles mixed-scale features natively.\n"
        "  - Provides feature importances, which are valuable for clinical interpretability.\n"
        "  - class_weight='balanced' up-weights the minority (distressed) class.\n\n"
        "Features are standardised with StandardScaler before training."
    )

    pdf.subsection_title("Evaluation metrics")
    pdf.body_text(
        "I evaluated the model on the held-out 20% test set:\n\n"
        "  Accuracy:    0.7027\n"
        "  ROC-AUC:     0.7330\n"
        "  Precision:   0.5909\n"
        "  Recall:      0.3514\n"
        "  F1 Score:    0.4407\n"
        "  Specificity: 0.8784\n\n"
        "Confusion matrix: TP=13, FP=9, FN=24, TN=65"
    )

    pdf.body_text(
        "Metric interpretation:\n"
        "- ROC-AUC of 0.73 indicates moderate discrimination ability, better than random.\n"
        "- High specificity (0.88) means the model correctly identifies most healthy cases.\n"
        "- Low recall (0.35) means the model misses many distressed cases. This is the "
        "main weakness and must be addressed for clinical use.\n"
        "- Precision of 0.59 means about 6 in 10 predicted-positive cases are truly "
        "distressed, which is acceptable for a screening tool.\n\n"
        "What the metrics do NOT tell us: ROC-AUC does not capture the clinical cost of "
        "different error types. In practice, missing a distressed baby (false negative) is "
        "far more dangerous than a false alarm (false positive). The threshold should be "
        "lowered to improve recall at the cost of some precision."
    )

    # ==================================================================
    # Section 3: Clinical Utility and Inference
    # ==================================================================
    pdf.add_page()
    pdf.section_title("3. Clinical Utility and Inference")

    pdf.subsection_title("Intended use")
    pdf.body_text(
        "This system is imagined as a decision-support tool that runs alongside the "
        "CTG monitor in a labour ward. It would NOT replace the clinician; instead it "
        "would provide an automated second opinion.\n\n"
        "Usage scenario:\n"
        "  1. The CTG monitor streams FHR and UC data continuously.\n"
        "  2. Every N minutes (e.g. every 10 min), the system extracts the last "
        "30 minutes of signal and computes the 29-feature vector.\n"
        "  3. The model produces a distress probability (0-1) and a binary alert.\n"
        "  4. If the probability exceeds a configurable threshold, a flag appears "
        "on the bedside display, prompting the nurse or doctor to review the trace.\n\n"
        "The threshold can be tuned to the clinical context: a lower threshold catches "
        "more true positives but raises more false alarms; a higher threshold reduces "
        "alarms but risks missing some cases."
    )

    pdf.subsection_title("generate_outcomes.py")
    pdf.body_text(
        "The inference script (generate_outcomes.py) implements the prediction pipeline:\n"
        "  - Input: either a raw WFDB record path or a pre-extracted .npz feature file.\n"
        "  - Processing: loads model.pkl and scaler.pkl from artifacts/, extracts features "
        "(if given a raw record), scales them, and runs the Random Forest.\n"
        "  - Output: for each input, prints the distress probability (float, 0-1) and "
        "the binary prediction (0=not distressed, 1=distressed).\n\n"
        "This mirrors the real-world deployment: the model receives a window of CTG data, "
        "computes features, and returns a probability. The clinical team acts on the alert."
    )

    # ==================================================================
    # Section 4: Limits and Next Steps
    # ==================================================================
    pdf.section_title("4. Limits and Next Steps")

    pdf.subsection_title("Known limitations")
    pdf.body_text(
        "1. Small dataset (552 recordings): limits model complexity and generalisability. "
        "Results may not transfer to other hospitals or populations without re-training.\n\n"
        "2. Retrospective labels: pH and Apgar are measured at delivery, but the model "
        "predicts from a 30-min window that may be well before delivery. The model is "
        "learning associations, not causal mechanisms.\n\n"
        "3. Low recall (0.35): the model misses a majority of distressed cases. This is "
        "the most critical limitation for clinical safety. Threshold tuning, SMOTE, or a "
        "different model architecture could improve this.\n\n"
        "4. No temporal modelling: the hand-crafted features summarise the window into "
        "a single vector, discarding the time-series structure. A sequential model "
        "(LSTM, Transformer) could capture progressive deterioration.\n\n"
        "5. Signal quality: some recordings have significant artefact (missing FHR). "
        "The current approach treats missing values as NaN and computes features from "
        "the valid portion, but this could bias estimates.\n\n"
        "6. No external validation: the model has only been tested on a random split of "
        "the same dataset. A proper external validation on a different hospital's data "
        "is essential before any clinical use."
    )

    pdf.subsection_title("Next steps with more time")
    pdf.body_text(
        "  - Hyperparameter tuning with cross-validation (grid search or Bayesian).\n"
        "  - Try gradient-boosted trees (XGBoost, LightGBM) which often outperform RF.\n"
        "  - Experiment with a 1-D CNN or LSTM on the raw 4 Hz signal.\n"
        "  - Sliding-window inference: produce a probability every 5-10 minutes to track "
        "how the risk evolves during labour (a risk trajectory).\n"
        "  - Calibration: use Platt scaling or isotonic regression so the output probability "
        "better reflects the true prevalence.\n"
        "  - SHAP values for per-prediction explanations.\n"
        "  - External validation on a second dataset."
    )

    pdf.output(output_path)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_report(os.path.join(script_dir, "report.pdf"))
