# Predictive Maintenance Agent for Manufacturing
## Cognitive AI Hackathon | GITAM Short-Listed Use Case #9

An intelligent, multi-stage Explainable AI (XAI) and root-cause reasoning system designed to eliminate unscheduled manufacturing downtime using the **AI4I 2020 Predictive Maintenance Benchmark Dataset**.

---

## 1. Project Overview & Problem Statement

Manufacturing equipment failures lead to catastrophic downtime, revenue loss, and safety hazards. Standard classification models only flag binary failure states without explaining *why* the failure occurred or *what action* the technician should take.

This system delivers:
1. **Anomaly & Failure Detection** via binary machine health modeling.
2. **Multi-Class Failure Diagnosis** classifying specific failure types (HDF, PWF, OSF, TWF, RNF).
3. **Transparent Explainable AI (XAI)** utilizing TreeExplainer SHAP attribution.
4. **Deterministic Physical Root-Cause Diagnosis** verifying mechanical power limits, thermal gradient collapse, and overstrain thresholds.
5. **Prescriptive Maintenance Action Protocols** generating step-by-step engineering directives.
6. **Unified Full-Stack Delivery** with a FastAPI backend service and an interactive Streamlit operations dashboard.

---

## 2. Team Division & Responsibilities

| Phase | Teammate | Assigned Responsibility | Core Deliverables |
|---|---|---|---|
| **Phase 1** | **Harshitha** | **Project Manager & Repository Lead** | Project milestone tracking, presentation deck, architecture design, and master repository documentation. |
| **Phase 2** | **Mallikarjun** | **Data Ingestion & EDA Engineer** | Dataset ingestion pipeline (`src/data/01_data_ingestion.py`), data validation, duplicate handling, and statistical EDA (`src/data/02_eda.py`). |
| **Phase 2** | **Srinidi** | **Feature Engineering Specialist** | Feature extraction (thermal differential, mechanical power), categorical encoding, and operational boundary checks (`src/preprocessing/preprocessor.py`). |
| **Phase 3** | **Jatin** | **Binary Classification Engineer** | Trained Random Forest binary classifier predicting `Machine Failure` vs `No Failure` with continuous probability (`models/binary_model.pkl`). |
| **Phase 3** | **Seenu** | **Multi-Class Classification Engineer** | Trained multi-class classifier identifying specific failure modes (HDF, PWF, OSF, TWF, RNF) (`models/multiclass_model.pkl`). |
| **Phase 4** | **Akhil** | **XAI & Agent Logic Developer** | TreeExplainer SHAP attribution (`src/xai/shap_explainer.py`), physics root-cause engine (`src/xai/root_cause_engine.py`), and unified agent orchestrator (`src/xai/agent.py`). |
| **Phase 5** | **Trusha** | **Backend API Engineer** | FastAPI REST API server (`src/api/app.py`), OpenAPI schemas (`src/api/schemas.py`), and health diagnostics. |
| **Phase 5** | **Neethu** | **Frontend UI & QA Lead** | Interactive Streamlit operations dashboard (`dashboard.py`), session-state management, live KPI cards, and automated QA test suite (`tests/`). |

---

## 3. Directory Structure

```text
cognizent/
├── data/
│   ├── ai4i2020.csv                          # AI4I 2020 benchmark dataset (10,000 records)
│   ├── clean_data.csv                        # Cleaned dataset
│   ├── train.csv                             # Stratified training split
│   └── test.csv                              # Stratified test split
├── docs/                                     # Reports, confusion matrices, and diagrams
├── models/
│   ├── binary_model.pkl                      # Jatin's binary failure classifier
│   └── multiclass_model.pkl                  # Seenu's multi-class failure classifier
├── src/
│   ├── __init__.py
│   ├── data/                                 # Mallikarjun & Srinidi (Phase 2)
│   │   ├── 01_data_ingestion.py
│   │   └── 02_eda.py
│   ├── preprocessing/                        # Srinidi (Phase 2)
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── models/                               # Jatin & Seenu (Phase 3)
│   │   ├── __init__.py
│   │   └── model_loader.py
│   ├── xai/                                  # Akhil (Phase 4)
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── root_cause_engine.py
│   │   ├── maintenance_rules.py
│   │   └── agent.py
│   └── api/                                  # Trusha (Phase 5)
│       ├── __init__.py
│       ├── schemas.py
│       └── app.py
├── tests/                                    # Neethu's QA Test Suite (Phase 5)
│   ├── test_agent.py
│   ├── test_api.py
│   ├── test_edge_cases.py
│   ├── test_frontend.py
│   ├── test_preprocessor.py
│   ├── test_rules.py
│   └── test_shap.py
├── dashboard.py                              # Neethu's Streamlit Dashboard (Phase 5)
├── demo.py                                   # End-to-end CLI validation script
├── requirements.txt                          # Project dependencies
└── README.md                                 # Master documentation
```

---

## 4. Root-Cause Reasoning Engine Logic (Physics Rules)

| Failure Mode | Physical Signals & Thresholds | Physical Root Cause | Prescriptive Maintenance Action |
|---|---|---|---|
| **Overstrain Failure (OSF)** | $\text{Tool Wear} \times \text{Torque} > \text{Threshold}$ ($L > 11000$, $M > 12000$, $H > 13000\text{ min}\cdot\text{Nm}$) | Excessive mechanical strain resulting from high operational torque on a degraded tool. | Immediately inspect tool condition and reduce mechanical load. Decrease feed rate and replace cutting insert. |
| **Heat Dissipation Failure (HDF)** | $(\text{Process Temp} - \text{Air Temp}) < 8.6\text{ K}$ and $\text{Speed} < 1380\text{ RPM}$ | Insufficient heat dissipation causing thermal buildup between spindle and workpiece. | Inspect cooling system and ventilation channels. Clear clogged heat fins and verify coolant flow rate. |
| **Power Failure (PWF)** | $\text{Power} = \tau \cdot \omega \notin [3500\text{ W}, 9000\text{ W}]$ | Drive power anomaly: motor overload ($>9000\text{ W}$) or motor stall ($<3500\text{ W}$). | Inspect motor power supply, electrical drive system, check VFD fault codes, and test spindle windings. |
| **Tool Wear Failure (TWF)** | $\text{Tool Wear} \ge 200\text{ min}$ | Severe cutting edge degradation and physical end-of-life tool wear exhaustion. | Replace tool insert immediately. Inspect tool holder for runout and recalibrate zero-point offsets. |
| **Random Failure (RNF)** | Process-independent stochastic defect | Stochastic disturbance or unmodeled external electrical/vibrational glitch. | Perform diagnostic sensor calibration and verify wiring; restart under observation. |
| **No Failure** | All parameters nominal | System operating within nominal operational parameters. | Continue standard operation. Maintain scheduled routine preventative inspections. |

---

## 5. Quickstart Guide

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Run Automated QA Tests
```powershell
pytest tests/ -v
```
*All 24 automated unit, integration, edge-case, and schema tests will execute and validate.*

### 3. Run the Streamlit Operations Dashboard (Neethu's UI)
```powershell
streamlit run dashboard.py
```
Open `http://localhost:8501` in your browser to access:
- **1-Click Test Presets** for all failure modes
- **Live Sensor Sliders** with dynamic health cards
- **SHAP Attribution Charts** and root cause analysis
- **Batch CSV Telemetry Inspection**

### 4. (Optional) Run the FastAPI REST Service (Trusha's Backend)
```powershell
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger Documentation available at `http://127.0.0.1:8000/docs`.
