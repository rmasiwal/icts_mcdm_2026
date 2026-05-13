"""
monsoon_onset/__init__.py
"""
from .loader import load_rainfall
from .detector import detect_onset, detect_first_wet_spell
from .climatology import compute_multi_year_onset, compute_climatological_onset
from .plotter import plot_climatological_onset, doy_to_date_string

__all__ = [
    "load_rainfall",
    "detect_onset",
    "detect_first_wet_spell",
    "compute_multi_year_onset",
    "compute_climatological_onset",
    "plot_climatological_onset",
    "doy_to_date_string",
]
