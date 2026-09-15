"""
Domain Maintenance Rules and Physical Formulae for AI4I 2020 Predictive Maintenance.

Encodes exact physical thresholds and engineering domain relationships:
- Overstrain Failure (OSF): Tool Wear * Torque thresholds by product variant.
- Heat Dissipation Failure (HDF): Temperature difference and rotational speed thresholds.
- Power Failure (PWF): Power (W) = Torque (Nm) * Speed (rad/s) boundaries [3500W, 9000W].
- Tool Wear Failure (TWF): Critical cumulative wear interval [200, 240] minutes.
- Random Failure (RNF): Stochastic / unmodeled component failure.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


# Physical thresholds from AI4I 2020 specification
OVERSTRAIN_THRESHOLDS = {
    "L": 11000.0,  # Tool Wear * Torque (min * Nm)
    "M": 12000.0,
    "H": 13000.0,
}

HDF_TEMP_DIFF_MAX = 8.6       # Process temp - Air temp (K)
HDF_SPEED_MAX = 1380.0        # Rotational speed (rpm)

PWF_POWER_MIN = 3500.0        # Power in Watts
PWF_POWER_MAX = 9000.0        # Power in Watts

TWF_WEAR_CRITICAL = 200.0     # Critical wear boundary in minutes


def compute_mechanical_power(torque_nm: float, speed_rpm: float) -> float:
    """Computes mechanical power in Watts: P = Torque * omega = Torque * (rpm * 2 * pi / 60)."""
    rad_per_sec = speed_rpm * (2.0 * math.pi / 60.0)
    return torque_nm * rad_per_sec


def evaluate_physical_signals(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates physical sensor readings against known physical failure criteria.

    Parameters
    ----------
    sensor_data : dict
        Contains 'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'.

    Returns
    -------
    dict with physical diagnostics, calculated values, and triggered condition flags.
    """
    product_type = str(sensor_data.get("Type", "L")).strip().upper()
    air_temp = float(sensor_data.get("Air temperature [K]", 300.0))
    proc_temp = float(sensor_data.get("Process temperature [K]", 310.0))
    speed = float(sensor_data.get("Rotational speed [rpm]", 1500.0))
    torque = float(sensor_data.get("Torque [Nm]", 40.0))
    tool_wear = float(sensor_data.get("Tool wear [min]", 0.0))

    temp_diff = proc_temp - air_temp
    power_watts = compute_mechanical_power(torque, speed)
    strain_product = tool_wear * torque
    osf_limit = OVERSTRAIN_THRESHOLDS.get(product_type, 11000.0)

    # Check physical conditions
    hdf_triggered = (temp_diff < HDF_TEMP_DIFF_MAX) and (speed < HDF_SPEED_MAX)
    pwf_triggered = (power_watts < PWF_POWER_MIN) or (power_watts > PWF_POWER_MAX)
    osf_triggered = strain_product > osf_limit
    twf_triggered = tool_wear >= TWF_WEAR_CRITICAL

    return {
        "temp_diff": round(temp_diff, 2),
        "power_watts": round(power_watts, 2),
        "strain_product": round(strain_product, 2),
        "osf_limit": osf_limit,
        "hdf_triggered": hdf_triggered,
        "pwf_triggered": pwf_triggered,
        "osf_triggered": osf_triggered,
        "twf_triggered": twf_triggered,
    }


def get_maintenance_rule(
    failure_type: str,
    physical_signals: Dict[str, Any],
    top_factors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """
    Maps failure classification and physical evidence to specific root causes and actionable recommendations.

    Parameters
    ----------
    failure_type : str
        Predicted failure type ('OSF', 'HDF', 'PWF', 'TWF', 'RNF', 'No Failure', or full names)
    physical_signals : dict
        Output from evaluate_physical_signals()
    top_factors : list of dicts, optional
        Top contributing factors from SHAP

    Returns
    -------
    dict with:
      - root_cause: Explanation of the underlying physical anomaly
      - recommendation: Prescriptive maintenance action
      - risk_level: 'Critical', 'High', 'Medium', or 'Low'
    """
    clean_type = failure_type.strip().upper()

    if "OVERSTRAIN" in clean_type or clean_type == "OSF":
        strain = physical_signals.get("strain_product", 0)
        limit = physical_signals.get("osf_limit", 11000)
        return {
            "root_cause": (
                f"Excessive mechanical overstrain ({strain:.0f} min*Nm vs {limit:.0f} limit) "
                "caused by high cutting torque applied to an already degraded tool."
            ),
            "recommendation": (
                "Immediately inspect tool condition and reduce excessive mechanical load. "
                "Decrease machine feed rate and replace the cutting insert to prevent spindle fracture."
            ),
            "risk_level": "High",
        }

    elif "HEAT" in clean_type or clean_type == "HDF":
        diff = physical_signals.get("temp_diff", 0)
        return {
            "root_cause": (
                f"Insufficient heat dissipation (thermal gradient only {diff:.1f} K below threshold of 8.6 K), "
                "leading to dangerous thermal build-up between spindle and workpiece."
            ),
            "recommendation": (
                "Inspect cooling system and ventilation channels immediately. "
                "Verify coolant flow rate, clean clogged heat dissipation fins, and ensure airflow is unobstructed."
            ),
            "risk_level": "High",
        }

    elif "POWER" in clean_type or clean_type == "PWF":
        p_watts = physical_signals.get("power_watts", 0)
        sub_cause = "Overload condition (> 9000 W)" if p_watts > PWF_POWER_MAX else "Motor stall / power drop (< 3500 W)"
        return {
            "root_cause": (
                f"Drive power anomaly ({p_watts:.0f} W outside safe operational range [3500 W, 9000 W]). "
                f"Indicates {sub_cause}."
            ),
            "recommendation": (
                "Inspect motor power supply, electrical drive system, and wiring connections. "
                "Check variable frequency drive (VFD) for fault codes and verify mechanical spindle free-rotation."
            ),
            "risk_level": "Critical",
        }

    elif "TOOL WEAR" in clean_type or clean_type == "TWF":
        return {
            "root_cause": (
                "Severe cutting edge degradation and physical end-of-life tool wear exhaustion "
                f"(tool wear reached {TWF_WEAR_CRITICAL:.0f}+ minutes)."
            ),
            "recommendation": (
                "Replace tool insert immediately. Inspect tool holder for runout, "
                "reset the tool wear accumulator, and recalibrate zero-point offsets."
            ),
            "risk_level": "High",
        }

    elif "RANDOM" in clean_type or clean_type == "RNF":
        return {
            "root_cause": (
                "Stochastic transient disturbance or unmodeled external mechanical vibration/power glitch."
            ),
            "recommendation": (
                "Perform diagnostic sensor calibration and check electrical ground wiring. "
                "Reset error state and monitor next operation cycle closely."
            ),
            "risk_level": "Medium",
        }

    else:
        # No Failure
        return {
            "root_cause": "System operating within nominal physiological and physical thresholds.",
            "recommendation": (
                "Continue standard operation. Maintain scheduled routine preventative inspections "
                "and maintain regular sensor logging."
            ),
            "risk_level": "Low",
        }
