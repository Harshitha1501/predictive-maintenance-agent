"""
Pydantic Schema Definitions for Predictive Maintenance REST API.
Guarantees clean, typed contracts for Trusha's FastAPI backend and Neethu's Streamlit UI.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SensorInput(BaseModel):
    """Input sensor payload for predictive maintenance inference."""

    type: str = Field(default="L", description="Product variant quality type ('L', 'M', or 'H')")
    air_temperature: float = Field(
        default=298.2,
        ge=250.0,
        le=350.0,
        alias="Air temperature [K]",
        description="Air temperature in Kelvin",
    )
    process_temperature: float = Field(
        default=308.6,
        ge=250.0,
        le=360.0,
        alias="Process temperature [K]",
        description="Process temperature in Kelvin",
    )
    rotational_speed: float = Field(
        default=1500.0,
        ge=100.0,
        le=5000.0,
        alias="Rotational speed [rpm]",
        description="Spindle rotational speed in RPM",
    )
    torque: float = Field(
        default=40.0,
        ge=0.0,
        le=200.0,
        alias="Torque [Nm]",
        description="Applied torque in Newton-meters",
    )
    tool_wear: float = Field(
        default=0.0,
        ge=0.0,
        le=500.0,
        alias="Tool wear [min]",
        description="Cumulative tool wear in minutes",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "type": "L",
                "air_temperature": 298.2,
                "process_temperature": 308.6,
                "rotational_speed": 1500.0,
                "torque": 40.0,
                "tool_wear": 120.0,
            }
        },
    )


class FactorImpact(BaseModel):
    """Individual contributing factor with risk impact rating."""

    feature: str = Field(description="Name of the physical sensor feature")
    impact: str = Field(description="Impact tier: 'High', 'Medium', or 'Low'")


class DetailedFactor(BaseModel):
    """Detailed feature attribution with numerical SHAP score and direction."""

    feature: str
    impact: str
    value: Any
    direction: str
    shap_value: float


class PhysicalTelemetry(BaseModel):
    """Physical equations evaluated by domain rules."""

    temperature_difference_k: float
    mechanical_power_watts: float
    overstrain_product_min_nm: float
    overstrain_threshold: float


class MaintenanceResponse(BaseModel):
    """
    Standardized response payload designed for Trusha's backend and Neethu's UI.
    """

    prediction: str = Field(description="Binary classification ('Machine Failure' or 'No Failure')")
    failure_probability: float = Field(description="Probability of failure between 0.0 and 1.0")
    failure_type: str = Field(description="Specific multi-class failure category")
    root_cause: str = Field(description="In-depth root cause explanation")
    risk_level: str = Field(description="'Critical', 'High', 'Medium', or 'Low'")
    top_factors: List[FactorImpact] = Field(description="Ranked top driving sensor features")
    recommendation: str = Field(description="Actionable prescriptive maintenance action")
    alert: bool = Field(description="True if risk_level is High or Critical")
    detailed_factors: Optional[List[DetailedFactor]] = None
    physical_telemetry: Optional[PhysicalTelemetry] = None
    all_class_probabilities: Optional[Dict[str, float]] = None
    warnings: Optional[List[str]] = None
    timestamp: str


class HealthCheckResponse(BaseModel):
    """API health and model availability status."""

    status: str
    binary_model_loaded: bool
    multiclass_model_loaded: bool
    shap_explainer_ready: bool
    timestamp: str
