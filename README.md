

# HiENBERT: Hierarchical Multi-Task Transformer for Multilingual Civic Complaint Classification

**HiENBERT (Hindi-English BERT)** is a hierarchical multi-task transformer framework for classifying multilingual civic complaints written in English, Hindi, and Hinglish.

This repository contains the experimental code, dataset-related resources, validation notebooks, visualizations, and analysis artifacts associated with the HiENBERT research project.

## Research Overview

Civic complaint platforms allow citizens to report public-service issues such as road damage, waste management problems, water leakage, sanitation concerns, and infrastructure failures.

Automated classification is challenging because of:

- Language heterogeneity across English, Hindi, and Hinglish.
- Code-mixing and Romanized Hindi expressions.
- Domain-specific civic terminology.
- Class imbalance in complaint categories.
- Hierarchical classification requirements.
- Severity assessment and prioritization.

HiENBERT addresses these challenges through a shared multilingual transformer representation and three task-specific prediction heads.

## Research Contributions

1. **HiENBERT framework:** A multilingual transformer architecture designed for civic complaint classification.
2. **Dual-backbone variants:** HiENBERT-Indic, based on IndicBERT, and HiENBERT-XLM, based on XLM-RoBERTa.
3. **Hierarchical multi-task prediction:** Simultaneous prediction of 8 primary categories, 24 subcategories, and 3 severity levels.
4. **CivicComp benchmark dataset:** A multilingual civic complaint dataset covering English, Hindi, and Hinglish.
5. **Synthetic data augmentation:** Additional synthetic complaints used to study data augmentation and class balancing.
6. **Imbalance-aware training:** Focal Loss, class weighting, and weighted random sampling.
7. **Explainable AI:** SHAP-based analysis of model decisions and feature contributions.

## HiENBERT Framework

### Model Variants

| Variant | Transformer Backbone |
|---|---|
| HiENBERT-Indic | IndicBERT |
| HiENBERT-XLM | XLM-RoBERTa-base |

### Architecture

The framework follows a shared-encoder, multi-task architecture:

1. Input multilingual complaint text.
2. Text preprocessing and normalization.
3. Multilingual tokenization using `[CLS]` and `[SEP]`.
4. Transformer-based contextual representation.
5. Shared representation layer.
6. Hierarchical feature layer.
7. Three task-specific prediction heads.
8. Joint multi-task optimization.

The three prediction tasks are:

- Primary category classification: 8 classes.
- Subcategory classification: 24 classes.
- Severity classification: Low, Medium, and High.

The severity prediction head additionally incorporates a sentiment feature derived through lexicon-based sentiment analysis.

### Conceptual Workflow

```text
English / Hindi / Hinglish Complaint
                  |
          Text Preprocessing
                  |
       Multilingual Tokenization
                  |
       Transformer Encoder
       (IndicBERT / XLM-R)
                  |
      Shared Representation Layer
                  |
       Hierarchical Feature Layer
                  |
       +----------+----------+
       |          |          |
   Primary     Subcategory  Severity
   Category    Prediction   Prediction
   (8 classes) (24 classes) (3 levels)
       |          |          |
       +----------+----------+
                  |
          Joint Multi-Task Loss
                  |
       Predictions and SHAP Analysis
```

## CivicComp Dataset

CivicComp is the multilingual civic complaint benchmark used in the study. The dataset is derived from the BBMP OpenCity platform and expanded with synthetic complaints generated using five large language models.

### Dataset Summary

| Characteristic | Value |
|---|---:|
| Languages | English, Hindi, Hinglish |
| Primary categories | 8 |
| Secondary categories | 24 |
| Severity levels | 3 |
| Original complaints | 16,000 |
| Synthetic complaints | 15,000 |
| Gross total before refinement | 31,000 |
| Unique records after category mapping and sampling | 20,236 |
| Multilingual expansion | 60,708 |

### Data Split

| Split | Percentage | Samples per language |
|---|---:|---:|
| Training | 70% | 14,165 |
| Validation | 15% | 3,035 |
| Testing | 15% | 3,036 |

### Severity Distribution

| Severity | Proportion |
|---|---:|
| High | 27.9% |
| Medium | 30.1% |
| Low | 42.0% |

### Dataset Quality Assessment

The manuscript reports human evaluation of synthetic data across English quality, Hindi quality, Hinglish quality, semantic validity, and translation consistency. Two annotators assessed the generated data, with weighted Cohen's kappa reported for inter-annotator agreement.

**Data access:** Please consult repository contents and applicable dataset terms before redistributing complaint records. Do not publish personal or sensitive information.

## Experimental Results

The accepted manuscript reports the following headline results for HiENBERT-XLM on the combined dataset:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 93.28% |
| Macro-F1 | 93.42% |
| Subcategory performance | 85.29% |
| Severity performance | 89.11% |
| Relative Macro-F1 improvement over imbalanced baseline | Up to 28.4% |

HiENBERT-XLM achieved the strongest subcategory and severity performance among the evaluated models.

Refer to the accepted paper for full comparison tables, per-language metrics, ablation studies, and evaluation protocols.

## Repository Structure

```text
HI-ENBERT/
│
├── Common_Dataset_Exp/
├── Original_Dataset_EXP/
├── Synthetic_Dataset_EXP/
├── dataset/
├── dataset_visuals/
├── figures/
│
├── CivicComp_Binary_Validation_Results.xlsx
├── Dataset_Validation.ipynb
├── Dataset_Visualization.ipynb
├── HIENBERT_Reviewer_Study.ipynb
├── Hinglish_Failure_Analysis.ipynb
├── Hinglish_Translation_Pipeline.ipynb
├── Result_Visualizations.ipynb
├── SHAP_Iterative.ipynb
│
├── checkpoint_hinglish.csv
├── hinglish_checkpoint.csv
├── hinglish_quality_issues.csv
├── complaints_data.json
├── complaint.png
│
├── complaint_classifier_app.py
├── complaint_classifier_app_final.py
├── requirements.txt
└── README.md
```

### Directory and File Description

| File / Directory | Purpose |
|---|---|
| `Common_Dataset_Exp/` | Combined dataset experiments |
| `Original_Dataset_EXP/` | Original dataset experiments |
| `Synthetic_Dataset_EXP/` | Synthetic dataset experiments |
| `dataset/` | Dataset resources |
| `dataset_visuals/` | Dataset visualizations |
| `figures/` | Research figures and visual assets |
| `Dataset_Validation.ipynb` | Dataset validation |
| `Dataset_Visualization.ipynb` | Dataset visualization |
| `HIENBERT_Reviewer_Study.ipynb` | Reviewer-oriented experimental analysis |
| `Hinglish_Failure_Analysis.ipynb` | Hinglish prediction error analysis |
| `Hinglish_Translation_Pipeline.ipynb` | Hinglish translation experiments |
| `Result_Visualizations.ipynb` | Experimental result visualizations |
| `SHAP_Iterative.ipynb` | SHAP explainability analysis |
| `complaint_classifier_app.py` | Complaint classification application |
| `complaint_classifier_app_final.py` | Final application script |

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TheShyamTripathi/HI-ENBERT.git
cd HI-ENBERT
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the relevant notebook from the repository.

## Running the Experiments

The repository contains separate experiment directories for:

- `Original_Dataset_EXP/`
- `Synthetic_Dataset_EXP/`
- `Common_Dataset_Exp/`

Recommended exploration order:

1. Review dataset resources and validation artifacts.
2. Inspect the original, synthetic, and combined dataset experiments.
3. Review experimental result visualizations.
4. Examine Hinglish failure analysis.
5. Inspect SHAP explainability analysis.
6. Review the complaint classification application.

Check the relevant notebooks and scripts for exact dataset paths, model checkpoints, package requirements, and execution order.

## Explainability and Error Analysis

The repository includes:

- SHAP analysis for feature contributions.
- Hinglish failure analysis for code-mixed text.
- Dataset validation and quality inspection.
- Experimental result visualization.

These analyses provide insights into model behavior beyond aggregate classification metrics.

## Reproducibility Notes

For reproducible experiments:

- Use the dataset version and split corresponding to the accepted manuscript.
- Keep training, validation, and test partitions separate.
- Do not use test data for training or hyperparameter selection.
- Record the selected transformer backbone and tokenizer.
- Preserve preprocessing, sequence-length, loss-function, and sampling configurations.
- Record random seeds, package versions, hardware, and checkpoint identifiers.
- Compare reproduced metrics against the evaluation protocol and tables in the accepted paper.

Intermediate checkpoints and analysis files may not correspond to the final reported experiment.

## Limitations

The study focuses on English, Hindi, and Hinglish civic complaints and the category structure defined in CivicComp. Performance on other languages, regions, complaint platforms, or substantially different writing styles requires separate evaluation.

Synthetic augmentation requires quality assessment and careful validation. Aggregate performance should also be considered alongside per-language performance and error analysis, particularly for code-mixed Hinglish complaints.

## Citation

**Paper title:**

*HiENBERT: A Hierarchical Multi-Task Transformer for Multilingual Civic Complaint Classification with Synthetic Data Augmentation and Explainable AI*

**Authors:** Shyam Tripathi, Kothamasu Hridayesh, and K. Ashwini.

```bibtex
@article{tripathi_hienbert,
  title   = {HiENBERT: A Hierarchical Multi-Task Transformer for
             Multilingual Civic Complaint Classification with
             Synthetic Data Augmentation and Explainable AI},
  author  = {Tripathi, Shyam and Kothamasu, Hridayesh and Ashwini, K.},
  journal = {IEEE Access},
  note    = {Accepted manuscript}
}
```

Add the final volume, page range, publication year, and DOI when the IEEE publication record is available.

## Acknowledgements

The authors acknowledge the support and research guidance provided by the Department of Computer Science and Engineering, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai, India.

The authors also acknowledge the BBMP OpenCity platform as the source of the original civic complaint data used in the study.

## License

No license is specified in this repository snapshot. Unless a license is added by the repository owner, reuse and redistribution of the source code and data are subject to applicable copyright and data-use restrictions.

---

**Research Areas:** Natural Language Processing · Multilingual NLP · Code-Mixed Language Processing · Transformer Models · Multi-Task Learning · Explainable AI · Civic Complaint Classification
