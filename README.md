# FastSpeech 2 Implementation for English and Telugu
### Academic Assignment – Text-to-Speech (TTS)

## Assignment Overview

This repository contains the implementation and experimental evaluation of the **FastSpeech 2 Text-to-Speech (TTS) model** for two languages: English and Telugu.

The assignment focuses on understanding and implementing the FastSpeech 2 architecture, preparing speech datasets, training language-specific models, generating speech samples, and analyzing the experimental results.

Two separate implementations are included:

1. **English TTS:** Implementation using the Svarah English speech dataset.
2. **Telugu TTS:** Implementation using the IndicTTS Telugu speech dataset.

## Assignment Objectives

- Understand the FastSpeech 2 non-autoregressive TTS architecture.
- Perform text and speech dataset preprocessing.
- Understand phoneme conversion and forced alignment.
- Implement duration, pitch, and energy prediction through the variance adaptor.
- Train separate English and Telugu FastSpeech 2 models.
- Generate speech samples from text inputs.
- Analyze training and validation loss.
- Visualize duration, pitch, and energy predictions.
- Document experimental observations and limitations.

## Technologies and Tools

- Python
- PyTorch
- FastSpeech 2
- NumPy and Pandas
- Librosa
- Matplotlib
- SoundFile
- Griffin-Lim reconstruction
- Git and Git LFS

## Repository Contents

- `English/` – English FastSpeech 2 implementation, configuration, inference, and trained checkpoint.
- `Telugu/` – Telugu FastSpeech 2 implementation, configuration, inference, and trained checkpoint.
- `Results/` – Generated speech samples, training graphs, validation graphs, and variance adaptor visualizations.
- `Report/Technical_Report.pdf` – Detailed assignment report.
- `.gitignore` and `.gitattributes` – Repository configuration.

## Datasets

**English – Svarah**
- 300 aligned utterances
- 270 training samples
- 30 validation samples

**Telugu – IndicTTS**
- 199 usable utterances
- 179 training samples
- 20 validation samples
- Approximately 1.27 hours of audio

Original datasets and intermediate preprocessing files are not included in this repository.

## FastSpeech 2 Architecture

The assignment implements the following processing pipeline:

```text
Input Text
    |
Text Normalization
    |
Phoneme Conversion
    |
FastSpeech 2 Encoder
    |
Variance Adaptor
(Duration, Pitch, Energy)
    |
Mel-Spectrogram Decoder
    |
Griffin-Lim Reconstruction
    |
Generated Speech (WAV)
```

## Experimental Results

Both language-specific models were trained for one epoch under the available assignment setup.

| Metric | English | Telugu |
|---|---:|---:|
| Training samples | 270 | 179 |
| Validation samples | 30 | 20 |
| Training loss | 7.1523 | 8.5027 |
| Validation loss | 5.1646 | 5.2605 |

Five generated speech samples for each language and their corresponding visualizations are included in `Results/`.

## Implementation and Execution

The repository contains separate training and inference scripts for English and Telugu.

**English training:**
```bash
python English/train_english.py
```

**Telugu training:**
```bash
python Telugu/train_telugu.py
```

**Inference scripts:**
- `English/inference.py`
- `Telugu/inference.py`

Training requires the corresponding datasets and preprocessing outputs, which are not included in this repository. Refer to the scripts and configuration files for the required inputs and dependencies.

## Experimental Limitations

- Only one training epoch was completed for each language.
- Relatively small dataset subsets were used.
- Griffin-Lim reconstruction was performed with 12 iterations.
- A dedicated neural vocoder was not integrated into the final experimental pipeline.
- MOS-based subjective speech quality evaluation was not performed.

These limitations are documented as part of the assignment evaluation.

## Assignment Report

The detailed technical report is available at:

`Report/Technical_Report.pdf`

It contains the methodology, implementation details, experimental observations, results, visualizations, and limitations.

