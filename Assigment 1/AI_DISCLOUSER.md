# AI Ethics Disclosure Appendix

**Student Name:** Shashank Satish Adsule  
**Course:** DA5402 – MLOps  
**Assignment:** A1 – Manual MLOps Challenge  
**Date:** February 14, 2026  

---

## 1. Overview of AI Usage

I used ChatGPT 5.2 selectively as a development assistant during this project.  
AI was primarily used for:

- Implementing Git commit tracking
- Improving console debugging output
- Structuring the Flask inference API
- Debugging monitoring and drift detection logic

AI was not used to replace conceptual understanding. All architectural decisions, debugging validation, and integration were performed independently.

---

## 2. Detailed Prompt Usage and Context

Below is a breakdown of the specific types of prompts I used and how they were applied.

---

### 2.1 Creating Git Commit Hash Code

**Prompt Type:**
> “How can I programmatically capture the current Git commit hash in Python?”

**Purpose:**
To ensure reproducibility by linking trained models to the exact code version used.

**Implementation:**
Added `get_git_commit_hash()` function in training pipeline:

Used in:
- `train.py` :contentReference[oaicite:3]{index=3}  
- Metadata saving section for model reproducibility  

**What I Did:**
- Integrated subprocess call
- Handled failure cases (no Git repo)
- Stored commit in model metadata
- Verified correct hash was saved during training

---

### 2.2 Stylized Print Statements for Better Debugging

**Prompt Type:**
> “How can I create visually structured terminal output for better debugging?”

**Purpose:**
To improve readability of pipeline stages and monitoring output.

**Implementation:**
Used ANSI styling and structured banners in:

- `train.py` :contentReference[oaicite:4]{index=4}  
- `inference.py` :contentReference[oaicite:5]{index=5}  
- `monitor.py` :contentReference[oaicite:6]{index=6}  

Examples:
- Section banners (`MODEL TRAINING PIPELINE`)
- Monitoring headers (`PRODUCTION MONITORING`)
- Clear stage-based logs (`[STEP 1]`, `[STEP 2]`, etc.)

**What I Did:**
- Adapted suggested formatting
- Ensured consistent style across scripts
- Verified outputs were readable during execution

---

### 2.3 Creating Server-Side Flask API Code

**Prompt Type:**
> “Help me structure a Flask API for ML inference with version tracking.”

**Purpose:**
To build a production-like inference layer supporting:
- Health checks
- Metadata exposure
- Prediction endpoints
- Deployment logging

**Implementation:**
Developed in:

- `inference.py` :contentReference[oaicite:7]{index=7}  

Features implemented:
- `/health`
- `/model-info`
- `/predict`
- `/batch-predict`
- Deployment logging
- Version loading logic

**What I Did Independently:**
- Integrated model version detection logic
- Connected config.yaml to inference behavior
- Implemented metadata exposure
- Debugged version mismatch issues
- Fixed prediction feature ordering logic
- Tested endpoints using manual API calls

---

### 2.4 Debugging Monitoring and Drift Detection

**Prompt Type:**
> “Why is my drift detection giving incorrect output?”
> “How should retraining triggers be structured?”

**Purpose:**
To validate statistical comparison logic and threshold-based retraining triggers.

**Implementation:**
Enhanced in:

- `monitor.py` :contentReference[oaicite:8]{index=8}  

Used for:
- Feature drift detection logic
- Metric comparison against training baseline
- Retraining trigger checks
- Monitoring report generation

**What I Did Independently:**
- Verified metric computations
- Checked correctness of threshold logic
- Identified version-loading inconsistencies
- Ensured production vs training comparisons were correct
- Manually tested monitoring pipeline end-to-end

---

## 3. Work Completed Independently

The following components were designed and integrated independently:

- Overall MLOps pipeline architecture
- Manual data versioning strategy
- Model version auto-increment logic
- Metadata registry logging
- Deployment tracking mechanism
- Production monitoring workflow
- Debugging cross-script version inconsistencies
- End-to-end system validation

All AI-generated suggestions were reviewed, modified where necessary, and tested.

---

## 4. Ethical Position

AI was used as a development assistant for:

- Boilerplate generation
- Debugging suggestions
- Formatting improvements

AI was not used to:
- Replace conceptual understanding
- Automatically design the entire pipeline
- Generate final reflective answers without comprehension

I confirm that:

- I understand all implemented components
- I can explain each file’s functionality
- I can modify and extend the system independently
- This disclosure accurately represents my AI usage

---

**Signed:**  
Shashank Satish Adsule  
**Date:** February 14, 2026  
