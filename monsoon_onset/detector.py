"""
monsoon_onset/detector.py
--------------------------
Onset detection algorithm driven by a YAML config.
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime


def detect_onset(rainfall_da: xr.DataArray,
                 threshold_da: xr.DataArray,
                 year: int,
                 cfg: dict) -> xr.DataArray:
    """
    Detect monsoon onset dates for a given year.

    The onset is defined as the first wet spell (first day > min_first_day_rain mm
    and N-day sum > threshold) starting from start_month/start_day that is NOT
    followed by a dry spell (M consecutive days with total < dry_spell_threshold mm)
    within dry_spell_check_days days.

    Parameters
    ----------
    rainfall_da : xr.DataArray
        Daily rainfall with dims [time, lat, lon] or [lat, lon, time].
        Must have standardized dimension names.
    threshold_da : xr.DataArray
        Wet-spell threshold values per grid point, dims [lat, lon].
    year : int
        Year being processed (used to set start date).
    cfg : dict
        Onset config dict from YAML, containing:
          - start_month          : int   (default 4)
          - start_day            : int   (default 1)
          - wet_spell_window     : int   (default 5)
          - dry_spell_window     : int   (default 10)
          - dry_spell_threshold  : float (default 5)
          - dry_spell_check_days : int   (default 30)
          - min_first_day_rain   : float (default 1)
          - mok                  : bool  (default False)

    Returns
    -------
    onset_da : xr.DataArray
        Onset dates (datetime64[ns]) per grid point, dims [lat, lon].
        NaT where no valid onset is found.
    """
    # --- Parse config --------------------------------------------------------
    start_month = cfg.get("start_month", 4)
    start_day_num = cfg.get("start_day", 1)
    wet_window = cfg.get("wet_spell_window", 5)
    dry_window = cfg.get("dry_spell_window", 10)
    dry_thresh = cfg.get("dry_spell_threshold", 5)
    dry_check = cfg.get("dry_spell_check_days", 30)
    min_rain = cfg.get("min_first_day_rain", 1)
    mok = cfg.get("mok", False)
    mok_date = datetime(year, 6, 2) if mok else None

    # --- Subset from start date ----------------------------------------------
    start_date = datetime(year, start_month, start_day_num)
    time_dates = pd.to_datetime(rainfall_da.time.values)
    start_candidates = np.where(time_dates >= start_date)[0]

    if len(start_candidates) == 0:
        print(f"Warning: Start date {start_date.strftime('%Y-%m-%d')} not found. "
              f"Using first available date.")
        start_idx = 0
    else:
        start_idx = start_candidates[0]

    print(f"Onset detection start: {time_dates[start_idx].strftime('%Y-%m-%d')}")
    rain = rainfall_da.isel(time=slice(start_idx, None))

    # --- Wet spell conditions ------------------------------------------------
    roll5 = rain.rolling(time=wet_window, min_periods=wet_window, center=False).sum()
    roll5_shifted = roll5.shift(time=-(wet_window - 1))
    wet_cond = (rain > min_rain) & (roll5_shifted > threshold_da)

    # --- Dry spell conditions ------------------------------------------------
    roll10 = rain.rolling(time=dry_window, min_periods=dry_window, center=False).sum()
    dry_cond = roll10 < dry_thresh

    # --- Per-gridpoint onset with dry-spell veto ----------------------------
    def _find_onset(wet_arr, dry_arr):
        wet_indices = np.where(wet_arr)[0]
        if len(wet_indices) == 0:
            return -1
        for oi in wet_indices:
            has_dry = False
            for ci in range(oi + 1, min(oi + dry_check + 1, len(dry_arr))):
                if dry_arr[ci]:
                    has_dry = True
                    break
            if not has_dry:
                # Apply MOK filter if enabled
                if mok:
                    onset_date = pd.to_datetime(rain.time.values[oi])
                    if onset_date.date() <= mok_date.date():
                        continue
                return oi
        return -1

    onset_indices = xr.apply_ufunc(
        _find_onset,
        wet_cond,
        dry_cond,
        input_core_dims=[["time"], ["time"]],
        output_dtypes=[int],
        vectorize=True,
    )

    # --- Convert indices to datetime ----------------------------------------
    # onset_indices has the same spatial dims as rain (minus 'time').
    # Transpose to canonical [lat, lon] order before extracting values.
    time_coords = rain.time.values

    # Ensure spatial dims are in [lat, lon] order
    spatial_dims = [d for d in onset_indices.dims]
    if spatial_dims != ["lat", "lon"]:
        onset_indices = onset_indices.transpose("lat", "lon")

    onset_array = np.full(onset_indices.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_mask = onset_indices.values >= 0

    for i in range(onset_indices.shape[0]):
        for j in range(onset_indices.shape[1]):
            if valid_mask[i, j]:
                idx = int(onset_indices.values[i, j])
                if 0 <= idx < len(time_coords):
                    onset_array[i, j] = time_coords[idx]

    onset_da = xr.DataArray(
        onset_array,
        coords=[("lat", onset_indices.lat.values), ("lon", onset_indices.lon.values)],
        name="onset_date",
        attrs={
            "description": "Onset date with dry-spell veto",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "wet_spell_window": wet_window,
            "dry_spell_window": dry_window,
            "dry_spell_threshold": dry_thresh,
            "dry_spell_check_days": dry_check,
            "mok_filter": mok,
        },
    )

    valid = (~pd.isna(onset_array)).sum()
    total = onset_array.size
    print(f"  Valid onsets: {valid}/{total} ({valid / total * 100:.1f}%)")

    return onset_da
