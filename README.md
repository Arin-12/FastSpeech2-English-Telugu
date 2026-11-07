# FastSpeech 2 Implementation for English and Telugu
### Technical Assignment Submission – Voice AI Models

**Submitted for:** Intern / Research Assistant Position  
**Organization:** Language Technologies Research Centre (LTRC), IIIT Hyderabad  
**Research Area:** Voice AI Models / Speech Processing  
**Submitted by:** Arin Jain

---

## Assignment Overview

This repository contains the implementation and experimental evaluation of the FastSpeech 2 Text-to-Speech (TTS) architecture for English and Telugu.

The assignment was provided as part of the initial evaluation process for the Intern / Research Assistant position at the Language Technologies Research Centre (LTRC), IIIT Hyderabad.

The work focuses on implementing language-specific speech synthesis pipelines, preparing speech datasets, training FastSpeech 2 models, generating speech samples, and analyzing experimental results.

### Implementations

1. **English Text-to-Speech:** FastSpeech 2 implementation using the Svarah English speech dataset.
2. **Telugu Text-to-Speech:** FastSpeech 2 implementation using the IndicTTS Telugu speech dataset.

## Assignment Objectives

- Understand the FastSpeech 2 architecture.
- Prepare and preprocess English and Telugu speech datasets.
- Perform phoneme processing and forced alignment.
- Implement duration, pitch, and energy prediction using the variance adaptor.
- Train language-specific FastSpeech 2 models.
- Generate speech waveforms from text inputs.
- Analyze training and validation losses.
- Visualize duration, pitch, and energy predictions.
- Document experimental results and limitations.

## Repository Contents

| Directory | Description |
|---|---|
| `English/` | English FastSpeech 2 implementation |
| `Telugu/` | Telugu FastSpeech 2 implementation |
| `Results/` | Generated speech samples, loss graphs, and variance adaptor visualizations |
| `Report/` | Detailed technical assignment report |

## Datasets

The implementation uses publicly available speech datasets from AI4Bharat and SPRINGLab. Both datasets contain substantially more samples than were used in this assignment. Due to the limited time available for completing the assignment, representative subsets were selected and processed for model training and evaluation.

### English – Svarah
- **Source:** [AI4Bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Full dataset size:** Approximately 6,656 utterances (about 9.6 hours of speech).
- **Subset used:** 300 aligned samples.
- **Training split:** 270 samples.
- **Validation split:** 30 samples.

### Telugu – IndicTTS
- **Source:** [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
- **Full dataset size:** Approximately 8,580 utterances.
- **Subset used:** 199 usable utterances.
- **Training split:** 179 samples.
- **Validation split:** 20 samples.

**Dataset selection note:** The complete datasets were not used for training. Only the stated subsets were selected and processed due to the assignment's time constraints. The reported training and validation results therefore correspond to these subsets, not to training on the full datasets.

Original datasets and intermediate preprocessing files are excluded from this public repository.

## Experimental Results

Both language-specific models were trained for one epoch under the available experimental setup.

| Evaluation Metric | English | Telugu |
|---|---:|---:|
| Training samples | 270 | 179 |
| Validation samples | 30 | 20 |
| Training loss | 7.1523 | 8.5027 |
| Validation loss | 5.1646 | 5.2605 |

Generated speech samples and training visualizations are available in the `Results/` directory.

## Technical Report

The detailed implementation methodology, experimental results, visualizations, and limitations are documented in:

`Report/Technical_Report.pdf`

## Experimental Limitations

- One training epoch was completed for each language.
- Relatively small dataset subsets were used.
- Griffin-Lim reconstruction was used with 12 iterations.
- A dedicated neural vocoder was not integrated into the final experimental pipeline.
- MOS-based subjective speech quality evaluation was not performed.

## Submission Details

**Assignment:** FastSpeech 2 – English and Telugu Text-to-Speech Implementation  
**Position:** Intern / Research Assistant – Voice AI Models  
**Organization:** Speech Processing Lab, LTRC, IIIT Hyderabad  
**Candidate:** Arin Jain

**GitHub Repository:**  
https://github.com/Arin-12/FastSpeech2-English-Telugu
