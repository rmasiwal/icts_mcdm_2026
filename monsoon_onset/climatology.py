"""
monsoon_onset/climatology.py
-----------------------------
Compute multi-year onset dataset and climatological mean onset.
"""

import numpy as np
import pandas as pd
import xarray as xr

from .loader import load_rainfall
from .detector import detect_onset


def compute_multi_year_onset(dataset_cfg: dict,
                              onset_cfg: dict,
                              years_cfg: dict,
                              threshold_da: xr.DataArray) -> xr.Dataset:
    """
    Compute onset dates for multiple years.

    Parameters
    ----------
    dataset_cfg : dict   - dataset section from YAML config
    onset_cfg   : dict   - onset section from YAML config
    years_cfg   : dict   - years section from YAML config
    threshold_da: xr.DataArray - pre-loaded threshold DataArray [lat, lon]

    Returns
    -------
    xr.Dataset with variable 'onset_date' and dims [year, lat, lon]
    """
    start_year = years_cfg.get("start", 1901)
    end_year = years_cfg.get("end", 2024)
    years = list(range(start_year, end_year + 1))

    print(f"\n{'='*60}")
    print(f"Processing {len(years)} years: {start_year}–{end_year}")
    print(f"Dataset : {dataset_cfg.get('name', 'unknown')}")
    print(f"{'='*60}\n")

    onset_arrays = []
    valid_years = []
    failed_years = []

    for idx, year in enumerate(years, 1):
        print(f"[{idx}/{len(years)}] Year {year} ...")
        try:
            rain = load_rainfall(year, dataset_cfg)
            onset_da = detect_onset(rain, threshold_da, year, onset_cfg)
            onset_arrays.append(onset_da.values)
            valid_years.append(year)
        except FileNotFoundError as e:
            print(f"  SKIP (file not found): {e}")
            failed_years.append(year)
        except Exception as e:
            print(f"  SKIP (error): {e}")
            failed_years.append(year)

    if not onset_arrays:
        raise ValueError("No years were successfully processed.")

    onset_3d = np.stack(onset_arrays, axis=0)   # [year, lat, lon]

    ds = xr.Dataset(
        {"onset_date": xr.DataArray(
            onset_3d,
            coords=[
                ("year", valid_years),
                ("lat", threshold_da.lat.values),
                ("lon", threshold_da.lon.values),
            ],
            name="onset_date",
            attrs={
                "description": "Multi-year monsoon onset dates",
                "dataset": dataset_cfg.get("name", "unknown"),
                "years_processed": str(valid_years),
                "failed_years": str(failed_years) if failed_years else "none",
            },
        )}
    )

    print(f"\nDone. Processed {len(valid_years)} years, "
          f"failed {len(failed_years)} years.")
    if failed_years:
        print(f"Failed years: {failed_years}")

    return ds


def compute_climatological_onset(onset_dataset: xr.Dataset,
                                  start_year: int = None,
                                  end_year: int = None):
    """
    Compute climatological mean onset DOY from a multi-year onset dataset.

    Parameters
    ----------
    onset_dataset : xr.Dataset  - output of compute_multi_year_onset
    start_year    : int, optional
    end_year      : int, optional

    Returns
    -------
    clim_doy        : xr.DataArray  - mean onset day-of-year [lat, lon]
    clim_onset_date : xr.DataArray  - mean onset date (ref year 2000) [lat, lon]
    valid_count     : xr.DataArray  - number of valid years per grid point
    """
    data = onset_dataset["onset_date"]

    if start_year is not None or end_year is not None:
        sy = start_year or int(data.year.min().values)
        ey = end_year or int(data.year.max().values)
        data = data.sel(year=slice(sy, ey))
        print(f"Climatology period: {sy}–{ey}")
    else:
        print("Climatology period: all available years")

    def _to_doy(dt_array):
        doy = np.full(dt_array.shape, np.nan)
        valid = ~pd.isna(dt_array)
        if valid.any():
            doy[valid] = pd.to_datetime(dt_array[valid]).dayofyear
        return doy

    doy_data = xr.apply_ufunc(
        _to_doy,
        data,
        input_core_dims=[["year"]],
        output_core_dims=[["year"]],
        vectorize=True,
    )

    clim_doy = doy_data.mean(dim="year", skipna=True)
    valid_count = (~np.isnan(doy_data)).sum(dim="year")

    clim_onset_date = xr.apply_ufunc(
        lambda doy: (np.datetime64("2000-01-01") + np.timedelta64(int(doy) - 1, "D")
                     if not np.isnan(doy) else np.datetime64("NaT")),
        clim_doy,
        vectorize=True,
    )

    clim_doy.attrs = {
        "description": "Climatological mean onset day of year",
        "units": "day of year",
        "total_years": int(data.year.size),
        "valid_years": f"{int(data.year.min().values)}–{int(data.year.max().values)}",
    }

    print(f"Mean onset DOY : {float(np.nanmean(clim_doy.values)):.1f}")
    print(f"Earliest DOY   : {float(np.nanmin(clim_doy.values)):.0f}")
    print(f"Latest DOY     : {float(np.nanmax(clim_doy.values)):.0f}")

    return clim_doy, clim_onset_date, valid_count
