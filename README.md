# CVIAF — Computer Vision Integrity Assurance Framework

**SIH 2026 Problem Statement ID 26228**
*Trustworthy Computer Vision Integrity Assurance for Data, Models and Inference Outputs in Multi-Contributor Pipelines*

**Organization:** Ministry of Defence (MoD) | **Department:** Indian Army (DGIS)
**Category:** Software | **Theme:** Blockchain & Cybersecurity

---

## Overview

CVIAF is an offline, air-gapped assurance framework for multi-contributor computer vision pipelines. It evaluates training data, trained models, and inference outputs to detect poisoning, substitution, tampering, replay, and distribution shift — without assuming any contributing source is trusted.

The complete workflow runs **offline** with **no cloud services or external APIs**.

---

## Core Capabilities

### 1. Training-Data Integrity
Detects label flipping, near-duplicate flooding, out-of-distribution (OOD) insertion, and simple trigger-patch backdoors. Aggregates sample-level evidence into a **source-level contributor risk score**.

### 2. Model Integrity
Computes model digest (SHA-256), performs behavioural fingerprinting against a reference battery, supports ONNX and PyTorch/TorchScript, and gracefully falls back when only black-box access is available.

### 3. Inference Provenance and Output Integrity
Cryptographically binds input image hash, model weight digest, preprocessing config, and output using:
- SHA-256 hashing
- Ed25519 signatures
- Sequence numbers, timestamps, nonces
- Hash-chained record linkage

Detects post-hoc alteration, replay, and substitution.

### 4. Distribution-Shift and Anomaly Assessment
Builds a reference distribution profile and computes a calibrated shift score (KS-based), distinguishing probable operational drift from suspicious manipulation where evidence supports it.

### 5. Analyst-Facing Assurance and Governance
Every flag contains:
- Human-readable reason
- Supporting evidence
- Confidence and severity
- Affected asset
- Recommended disposition (`accept` / `review` / `quarantine`)

Maintains a tamper-evident audit trail (hash-chained JSONL) and produces a Merkle-rooted assurance report.

---

## Tech Stack

- **Language:** Python 3.10+
- **Computer Vision:** OpenCV, PIL, imagehash
- **ML Runtime:** PyTorch, TorchScript, ONNX Runtime
- **Analysis:** scikit-learn, NumPy, pandas, SciPy
- **Cryptography:** `cryptography` (Ed25519, SHA-256)
- **Storage:** JSON, JSONL, SQLite
- **UI:** Streamlit (local dashboard)
- **CLI:** argparse

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cviaf-sih-2026.git
cd cviaf-sih-2026

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

