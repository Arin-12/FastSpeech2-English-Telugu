from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "Results"
OUT = ROOT / "Report" / "Technical_Report.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"],
    fontSize=23, leading=30, alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"), spaceAfter=18
))
styles.add(ParagraphStyle(
    name="Section", parent=styles["Heading1"],
    fontSize=16, leading=20, textColor=colors.HexColor("#17365D"),
    spaceBefore=12, spaceAfter=8
))
styles.add(ParagraphStyle(
    name="SubSection", parent=styles["Heading2"],
    fontSize=12, leading=15, textColor=colors.HexColor("#24527A"),
    spaceBefore=8, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="BodySmall", parent=styles["BodyText"],
    fontSize=9, leading=13, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="CaptionSmall", parent=styles["BodyText"],
    fontSize=8, leading=10, alignment=TA_CENTER,
    textColor=colors.grey, spaceAfter=8
))

doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    rightMargin=0.62*inch, leftMargin=0.62*inch,
    topMargin=0.65*inch, bottomMargin=0.65*inch,
    title="FastSpeech 2 English and Telugu TTS - Technical Report",
    author="Arin Jain"
)

story = []

def P(text, style="BodySmall"):
    return Paragraph(text, styles[style])

def section(title):
    story.append(P(title, "Section"))

def subsection(title):
    story.append(P(title, "SubSection"))

def table(data, widths=None):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("LEADING", (0,0), (-1,-1), 11),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#B7C9D6")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

def add_image(path, caption, width=6.7*inch, height=3.0*inch):
    path = Path(path)
    if path.exists():
        img = Image(str(path))
        ratio = min(width / img.imageWidth, height / img.imageHeight)
        img.drawWidth = img.imageWidth * ratio
        img.drawHeight = img.imageHeight * ratio
        story.append(img)
        story.append(P(caption, "CaptionSmall"))
    else:
        story.append(P("Plot not found: " + str(path), "CaptionSmall"))

def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(0.62*inch, 0.35*inch, "FastSpeech 2 English and Telugu TTS")
    canvas.drawRightString(7.85*inch, 0.35*inch, f"Page {document.page}")
    canvas.restoreState()

# COVER
story.append(Spacer(1, 1.1*inch))
story.append(P("TECHNICAL REPORT", "CoverTitle"))
story.append(P(
    "Implementation and Evaluation of FastSpeech 2 "
    "for English and Telugu Text-to-Speech Synthesis",
    "Title"
))
story.append(Spacer(1, 0.35*inch))
story.append(P("<b>Submitted by:</b> Arin Jain", "BodyText"))
story.append(P("<b>Project:</b> FastSpeech 2 Multilingual TTS Implementation", "BodyText"))
story.append(P("<b>Languages:</b> English and Telugu", "BodyText"))
story.append(P("<b>Submission date:</b> 7 November 2026", "BodyText"))
story.append(Spacer(1, 0.5*inch))
story.append(P(
    "This report documents the implemented pipelines, experimental setup, "
    "training results, speech generation outputs, and limitations for two "
    "separate FastSpeech 2 text-to-speech systems.",
    "BodySmall"
))
story.append(PageBreak())

# ABSTRACT
section("Abstract")
story.append(P(
    "This project implements separate FastSpeech 2 text-to-speech (TTS) "
    "pipelines for English and Telugu. The systems use language-specific "
    "text processing, acoustic feature preparation, duration supervision, "
    "pitch and energy features, model training, and inference. Five test "
    "utterances were synthesized for each language. Duration, pitch, and "
    "energy visualizations were generated for every utterance. Griffin-Lim "
    "was used for waveform reconstruction from predicted mel spectrograms. "
    "The experiments demonstrate an end-to-end implementation, while the "
    "limited training subset and short training schedule are important "
    "constraints on the quality and generalization of the generated speech."
))
section("1. Introduction and Objectives")
story.append(P(
    "Text-to-speech synthesis converts written text into spoken audio. "
    "FastSpeech 2 is a non-autoregressive acoustic model that predicts "
    "mel-spectrogram representations from input text. Its variance adaptor "
    "models duration, pitch, and energy to provide explicit control over "
    "prosodic characteristics."
))
story.append(P("<b>Project objectives</b>"))
for x in [
    "Implement separate FastSpeech 2 pipelines for English and Telugu.",
    "Prepare language-specific text and acoustic features.",
    "Train and validate each model using separate data splits.",
    "Generate speech for five test utterances per language.",
    "Visualize predicted duration, pitch, and energy for each test utterance.",
    "Compare the experimental outcomes and document limitations."
]:
    story.append(P("• " + x))

section("2. FastSpeech 2 Architecture")
story.append(P(
    "The implemented architecture follows the FastSpeech 2 structure. "
    "A text encoder converts token sequences into hidden representations. "
    "The variance adaptor adds duration, pitch, and energy information. "
    "The decoder transforms the conditioned representations into mel "
    "spectrogram frames. A waveform reconstruction stage converts the "
    "predicted mel representation into audio."
))
table([
    ["Component", "Function"],
    ["Text encoder", "Encodes input phoneme/character tokens into hidden representations."],
    ["Duration predictor", "Predicts the number of acoustic frames associated with each input token."],
    ["Pitch predictor", "Models fundamental-frequency-related information."],
    ["Energy predictor", "Models frame-level acoustic energy."],
    ["Variance adaptor", "Combines duration, pitch, and energy information with encoder outputs."],
    ["Mel decoder", "Predicts the output mel spectrogram."],
    ["Vocoder / reconstruction", "Converts the mel spectrogram into a waveform; Griffin-Lim was used here."]
])
story.append(P(
    "The two language implementations use separate model directories, "
    "configurations, preprocessing artifacts, checkpoints, and inference outputs."
))

section("3. Dataset and Experimental Setup")
table([
    ["Item", "English", "Telugu"],
    ["Dataset", "Svarah", "SPRINGLab IndicTTS Telugu"],
    ["Source split", "4,662 train / 518 validation records", "199 usable aligned utterances"],
    ["Source duration", "7.18 h train / 0.79 h validation", "Approximately 1.27 h usable audio"],
    ["Prepared model subset", "300 aligned samples", "199 aligned samples"],
    ["Training subset", "270", "179"],
    ["Validation subset", "30", "20"],
    ["Training epochs completed", "1", "1"],
    ["Training loss", "7.1523", "8.5027"],
    ["Validation loss", "5.1646", "5.2605"],
    ["Compute", "CPU", "CPU"]
])
story.append(P(
    "<b>Dataset scope clarification:</b> The English source split contains "
    "approximately 7.18 hours of training audio and 0.79 hours of validation "
    "audio, but only 300 aligned samples were prepared and used for the "
    "reported model run. The Telugu experiment used 199 usable aligned "
    "utterances, totaling approximately 1.27 hours. These experiments do "
    "not meet the originally specified five-hour training and one-hour "
    "validation target for each language."
))

section("4. English Pipeline")
story.append(P(
    "The English pipeline uses the Svarah dataset. Audio was processed into "
    "mono acoustic inputs and resampled to 22,050 Hz for feature preparation. "
    "The mel representation uses 80 mel channels, an FFT size of 1,024, "
    "a hop length of 256, and an 8,000 Hz upper mel-frequency limit."
))
story.append(P(
    "Text normalization expands numbers and abbreviations, normalizes "
    "text to the model's supported uppercase symbol inventory, and "
    "represents word boundaries using the configured separator. A pretrained "
    "English wav2vec 2.0 CTC model was used to obtain alignments. These "
    "alignments provide duration supervision. WORLD-based F0 extraction "
    "and RMS energy features were prepared for the variance adaptor."
))
story.append(P(
    "The English model was trained for one completed epoch using 270 "
    "training examples and evaluated on 30 validation examples. The "
    "resulting training loss was 7.1523 and validation loss was 5.1646."
))
subsection("English training curves")
add_image(RESULTS / "English" / "training_loss.png", "Figure 1. English training loss.")
add_image(RESULTS / "English" / "validation_loss.png", "Figure 2. English validation loss.")

section("5. Telugu Pipeline")
story.append(P(
    "The Telugu pipeline uses the SPRINGLab IndicTTS Telugu dataset. "
    "A Telugu wav2vec 2.0 CTC alignment model was used to obtain token "
    "alignment information. Audio was processed at 16,000 Hz with an "
    "80-channel mel representation, FFT/window size 1,024, hop length "
    "256, and a 7,600 Hz upper mel-frequency limit."
))
story.append(P(
    "Telugu-specific preprocessing includes script-aware text handling, "
    "CTC alignment, character-level duration allocation, WORLD-based "
    "pitch extraction, and energy feature preparation. One utterance was "
    "excluded because it contained a character unsupported by the configured "
    "aligner vocabulary. The resulting split contained 179 training and "
    "20 validation utterances."
))
story.append(P(
    "The Telugu model completed one epoch with training loss 8.5027 and "
    "validation loss 5.2605."
))
story.append(P(
    "<b>Text normalization note:</b> Some Telugu synthesis inputs were "
    "normalized to accommodate the available vocabulary and aligner. "
    "In particular, an unsupported character in T1 led to replacement of "
    "the original word with a simpler word; T3 spelling was corrected, "
    "and numerals in T4 were expanded into Telugu words. The original "
    "assignment text and the actual synthesis input are retained in "
    "inference metadata."
))
subsection("Telugu training curves")
add_image(RESULTS / "Telugu" / "training_loss.png", "Figure 3. Telugu training loss.")
add_image(RESULTS / "Telugu" / "validation_loss.png", "Figure 4. Telugu validation loss.")

section("6. Speech Generation and Test Samples")
story.append(P(
    "Inference was performed using the best saved checkpoint for each "
    "language. Five utterances were generated for English (E1, E2a, E2b, "
    "E3, E4) and five for Telugu (T1, T2a, T2b, T3, T4). The generated "
    "waveforms were reconstructed from predicted mel spectrograms using "
    "the Griffin-Lim algorithm with 12 iterations."
))
table([
    ["Language", "Sample IDs", "Generated audio location"],
    ["English", "E1, E2a, E2b, E3, E4", "English/outputs/english/test_generated_audios/"],
    ["Telugu", "T1, T2a, T2b, T3, T4", "Telugu/outputs/telugu/test_generated_audios/"]
])
story.append(P(
    "The English test set includes a long technical word, the two different "
    "uses of 'conduct', an intentionally unusual 'rhyth m' spelling, and "
    "a sentence containing a time, date, and proper name. The Telugu set "
    "contains the supplied test items, with the documented normalization "
    "changes described in Section 5."
))

section("7. Variance Adaptor Analysis")
story.append(P(
    "For every generated utterance, three plots were saved. Duration plots "
    "show predicted acoustic-frame allocation across input tokens. Pitch "
    "plots show the predicted pitch trajectory, and energy plots show "
    "the predicted energy trajectory over mel frames. These visualizations "
    "allow inspection of the model's predicted temporal and prosodic patterns. "
    "They are model predictions and should not be interpreted as objective "
    "measurements of naturalness or intelligibility."
))

for lang, ids in [
    ("English", ["E1", "E2a", "E2b", "E3", "E4"]),
    ("Telugu", ["T1", "T2a", "T2b", "T3", "T4"])
]:
    section(lang + " variance-adaptor plots")
    base = RESULTS / lang / "variance_adaptor_plots"
    for sid in ids:
        subsection(lang + " - " + sid)
        add_image(base / "duration" / f"{sid}_duration.png",
                  f"{sid}: predicted duration.")
        add_image(base / "pitch" / f"{sid}_pitch.png",
                  f"{sid}: predicted pitch.")
        add_image(base / "energy" / f"{sid}_energy.png",
                  f"{sid}: predicted energy.")

section("8. Comparative Discussion")
table([
    ["Aspect", "English", "Telugu"],
    ["Input representation", "Uppercase normalized English symbols", "Telugu script and configured character vocabulary"],
    ["Alignment", "English wav2vec 2.0 CTC", "Telugu wav2vec 2.0 CTC"],
    ["Training examples", "270", "179"],
    ["Validation examples", "30", "20"],
    ["Completed epochs", "1", "1"],
    ["Training loss", "7.1523", "8.5027"],
    ["Validation loss", "5.1646", "5.2605"],
    ["Waveform reconstruction", "Griffin-Lim", "Griffin-Lim"]
])
story.append(P(
    "Both language pipelines completed feature preparation, one training "
    "epoch, checkpoint saving, and five-sentence inference. The loss values "
    "are not directly comparable measures of language quality because the "
    "datasets, token inventories, feature distributions, and training "
    "subsets differ."
))

section("9. Limitations")
for x in [
    "Training was limited to one completed epoch per language because of the submission time constraint.",
    "Only a subset of the English aligned dataset was used for the reported training run.",
    "The Telugu dataset contained approximately 1.27 hours of usable aligned audio, below the originally requested duration.",
    "The models were trained on CPU with small batches and limited examples.",
    "Griffin-Lim reconstruction was used instead of a separately trained neural vocoder; this can reduce waveform naturalness and clarity.",
    "Telugu vocabulary and alignment constraints required text normalization for some test sentences.",
    "No formal human listening study, MOS evaluation, or objective intelligibility benchmark was conducted."
]:
    story.append(P("• " + x))

section("10. Reproducibility and Project Artifacts")
story.append(P(
    "The project contains separate English and Telugu model code/configuration "
    "directories, preprocessing scripts, training scripts, inference scripts, "
    "saved checkpoints, generated audio, plots, and metadata. Python 3.11 "
    "was used in a virtual environment. The experiments were executed on CPU."
))
table([
    ["Artifact", "Location"],
    ["English training", "English/train_english.py"],
    ["English inference", "English/inference.py"],
    ["English checkpoint", "English/outputs/english/best_model.pth"],
    ["Telugu training", "Telugu/train_telugu.py"],
    ["Telugu inference", "Telugu/inference.py"],
    ["Telugu checkpoint", "Telugu/outputs/telugu/best_model.pth"],
    ["English results", "Results/English/"],
    ["Telugu results", "Results/Telugu/"]
])
story.append(P(
    "The checkpoints and configuration files should be retained together. "
    "For exact reproduction, the corresponding datasets, preprocessing "
    "dependencies, aligner models, and model configurations are also required."
))

section("11. Conclusion")
story.append(P(
    "Separate FastSpeech 2 TTS pipelines for English and Telugu were "
    "implemented and exercised through training and inference. Both systems "
    "produced five test audio samples and the associated duration, pitch, "
    "and energy plots. Training and validation loss curves and model "
    "checkpoints were saved. The work demonstrates an end-to-end experimental "
    "implementation, while limited data preparation, one-epoch training, "
    "and Griffin-Lim reconstruction remain important limitations. Further "
    "training on the full target datasets and evaluation with a neural "
    "vocoder and listening tests would be necessary for a more robust "
    "quality assessment."
))

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("REPORT CREATED:", OUT)
print("SIZE (MB):", round(OUT.stat().st_size / (1024*1024), 2))
