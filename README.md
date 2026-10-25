# Toxic Comment & Unintended Bias Classification Dashboard

An end-to-end Machine Learning, REST API, and Angular web application with automated CI/CD and GitHub Pages deployment.

Based on the research methodology presented in:
> **Khan, Md Azim (2023).** *Determination of toxic comments and unintended model bias minimization using Deep learning approach.* Department of Information Systems, University of Maryland, Baltimore County (UMBC). [arXiv:2311.04789v1](https://arxiv.org/abs/2311.04789).

---

## 1. Project Architecture

```
Browser / Angular Single Page UI (http://localhost:4200)
      │
      │  POST /api/predict (JSON payload)
      ▼
Angular Proxy (frontend/proxy.conf.json)  ──>  Flask API (http://127.0.0.1:5000)
                                                    │
                                                    ▼
                                          Input Schema Validation
                                                    │
                                                    ▼
                                          Fitted TF-IDF Preprocessor
                                                    │
                                                    ▼
                                          Balanced Logistic Regression Model
                                                    │
                                                    ▼
                                          JSON Prediction & Probabilities
```

* **Dataset**: Jigsaw Unintended Bias in Toxicity Classification (Conversation AI / Jigsaw).
* **Task**: Binary Classification (`0: Non-Toxic`, `1: Toxic`). Decision boundary: toxicity $\ge 0.5$.
* **Features**: `comment_text` (raw text input).
* **Preprocessing**: Lowercasing, punctuation and symbol filtering, stop words handling, and sublinear TF-IDF vectorization (`ngram_range=(1, 2)`).
* **Model**: Logistic Regression with `class_weight='balanced'` and $L_2$ regularization.
* **Packaging**: Standard `sklearn.pipeline.Pipeline` bundle exported to `backend/models/model_bundle.joblib`.

---

## 2. Directory Structure

```
├── .github/workflows/
│   ├── ci.yml                     # Backend pytest + Frontend build CI
│   └── pages.yml                  # Automatic GitHub Pages deployment
├── backend/
│   ├── models/
│   │   └── model_bundle.joblib    # Serialized preprocessor + model bundle
│   ├── tests/
│   │   ├── test_structure.py      # Repository structure smoke tests
│   │   └── test_api.py            # API health & prediction test suite
│   ├── app.py                     # Flask REST API (/health, /predict)
│   ├── train.py                   # Preprocessing & training pipeline script
│   ├── make_payload.py            # Generates valid sample test payload
│   ├── test_pipeline.py           # Independent local inference test
│   ├── test_api.py                # Direct HTTP request test script
│   ├── test_payload.json          # 1-row valid input JSON
│   └── requirements.txt           # Python backend dependencies
├── data/
│   ├── jigsaw_toxicity_dataset.csv# Curated toxicity and identity dataset
│   └── sample_inputs.csv          # Single-row non-sensitive test sample
├── frontend/                      # Standalone Angular application
│   ├── src/app/
│   │   ├── app.ts                 # Dashboard component logic
│   │   ├── app.html               # Responsive form & results UI
│   │   ├── app.css                # Polished dark-mode styles
│   │   ├── feature-schema.ts      # Model input feature specifications
│   │   └── prediction.service.ts  # Angular HTTP client service
│   ├── proxy.conf.json            # Dev proxy mapping /api to Flask
│   ├── angular.json
│   └── package.json
├── .gitignore
└── README.md
```

---

## 3. Local Setup & Execution Guide

### Prerequisites
* Python 3.10+ (tested on Python 3.12 / 3.14)
* Node.js v20+ & npm

### Backend Installation & Testing
```bash
# 1. Install dependencies
pip install -r backend/requirements.txt

# 2. Run independent model inference test
python backend/test_pipeline.py

# 3. Run automated pytest test suite
python -m pytest -q backend/tests
```

### Running the End-to-End Application

Open two terminals:

#### Terminal 1 — Start Flask Backend
```bash
python backend/app.py
# Server runs on http://127.0.0.1:5000
```
Verify the health endpoint:
```bash
curl -i http://127.0.0.1:5000/health
```

#### Terminal 2 — Start Angular Frontend
```bash
cd frontend
npm install
npm start -- --proxy-config proxy.conf.json
# Or: npx ng serve --proxy-config proxy.conf.json
```
Open **http://localhost:4200** in your browser.

---

## 4. Continuous Integration & Deployment (CI/CD)

1. **Continuous Integration (`.github/workflows/ci.yml`)**:
   * Triggers on `push` and `pull_request` to `main`.
   * Runs Python backend structure tests and API test cases.
   * Compiles the Angular frontend and produces build artifacts.

2. **Continuous Deployment (`.github/workflows/pages.yml`)**:
   * Triggers on `push` to `main` and `workflow_dispatch`.
   * Builds the production Angular bundle with the repository's `base-href`.
   * Deploys the static interface to **GitHub Pages**.

> **Note on Static Deployment Boundary**:
> GitHub Pages hosts the static client application. Live machine learning predictions require the local Flask REST API or a hosted server. The web UI automatically detects static deployment environments and displays an informational banner.

---

## 5. Research & Ethics Disclaimer
This system is an academic research and educational prototype based on the Jigsaw Unintended Bias in Toxicity Classification benchmark and the findings of Khan (2023). It is not intended for automated moderation without human review.
