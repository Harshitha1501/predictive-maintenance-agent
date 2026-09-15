"""
FastAPI Application for Predictive Maintenance XAI & Maintenance Agent.
Designed for immediate consumption by Trusha's backend and Neethu's Streamlit dashboard.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import HealthCheckResponse, MaintenanceResponse, SensorInput
from src.xai.agent import PredictiveMaintenanceAgent

# Global agent singleton
_agent_instance: PredictiveMaintenanceAgent = None


def get_agent() -> PredictiveMaintenanceAgent:
    """Returns the singleton instance of the PredictiveMaintenanceAgent."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PredictiveMaintenanceAgent()
    return _agent_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and SHAP explainers on server startup."""
    get_agent()
    yield


app = FastAPI(
    title="Predictive Maintenance XAI & Agent API",
    description=(
        "Intelligent reasoning and explainable AI system for manufacturing predictive maintenance. "
        "Provides binary failure prediction, multi-class failure typing, SHAP attribution, "
        "physics-based root-cause diagnosis, and prescriptive maintenance actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for dashboard and frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """API Landing Page with documentation links."""
    return {
        "service": "Predictive Maintenance XAI & Reasoning Agent API",
        "status": "online",
        "documentation": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "predict_and_explain": "POST /api/v1/predict",
            "health_check": "GET /api/v1/health",
            "sample_cases": "GET /api/v1/sample-cases",
        },
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Diagnostics"])
def health_check():
    """Returns system status and model readiness."""
    agent = get_agent()
    b_ready = agent.model_loader.binary_model is not None
    m_ready = agent.model_loader.multiclass_model is not None
    shap_ready = agent.binary_shap is not None or agent.multiclass_shap is not None

    return HealthCheckResponse(
        status="healthy" if (b_ready or m_ready) else "degraded",
        binary_model_loaded=b_ready,
        multiclass_model_loaded=m_ready,
        shap_explainer_ready=shap_ready,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.post("/api/v1/predict", response_model=MaintenanceResponse, tags=["Inference"])
def predict_and_explain(sensor_input: SensorInput):
    """
    Primary API endpoint for Trusha's backend.
    Accepts raw sensor telemetry and executes the full agent reasoning chain:
    Prediction -> Multi-Class Type -> SHAP Top Factors -> Root Cause -> Action Recommendation.
    """
    agent = get_agent()
    try:
        # Convert Pydantic model to dictionary compatible with preprocessor
        input_data = sensor_input.model_dump(by_alias=True)
        result = agent.analyze(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/api/v1/sample-cases", tags=["Testing"])
def get_sample_cases():
    """
    Returns verified sample telemetry payloads representing each failure mode
    for convenient 1-click testing and demonstration in the dashboard.
    """
    return {
        "normal_operating_condition": {
            "label": "Healthy Machine (Nominal Operation)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 12.0,
            },
        },
        "overstrain_failure_osf": {
            "label": "Overstrain Failure (High Torque + Worn Tool)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.4,
                "process_temperature": 308.2,
                "rotational_speed": 1282.0,
                "torque": 60.7,
                "tool_wear": 216.0,
            },
        },
        "heat_dissipation_failure_hdf": {
            "label": "Heat Dissipation Failure (Thermal Gradient Collapse)",
            "telemetry": {
                "type": "M",
                "air_temperature": 302.4,
                "process_temperature": 310.2,
                "rotational_speed": 1332.0,
                "torque": 52.3,
                "tool_wear": 142.0,
            },
        },
        "power_failure_pwf": {
            "label": "Power Failure (Drive Overload / Stall)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.9,
                "process_temperature": 309.1,
                "rotational_speed": 2861.0,
                "torque": 4.6,
                "tool_wear": 143.0,
            },
        },
        "tool_wear_failure_twf": {
            "label": "Tool Wear Failure (Exhausted Tool Life)",
            "telemetry": {
                "type": "L",
                "air_temperature": 298.8,
                "process_temperature": 308.9,
                "rotational_speed": 1455.0,
                "torque": 41.3,
                "tool_wear": 208.0,
            },
        },
    }
