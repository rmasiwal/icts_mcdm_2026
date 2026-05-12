"""
monsoon_onset/loader.py
-----------------------
Dataset-agnostic rainfall loader driven by a YAML config.
"""

import os
import xarray as xr


def load_rainfall(year: int, cfg: dict) -> xr.DataArray:
    """
    Load daily rainfall data for a given year using dataset config.

    Parameters
    ----------
    year : int
        Year to load.
    cfg : dict
        Dataset config dict (parsed from YAML), containing keys:
          - data_folder      : str  - directory containing NetCDF files
          - file_pattern     : str  - filename pattern with {year} placeholder
          - precip_var       : str  - variable name in the NetCDF file
          - dim_mapping      : dict - dimension renames {original: standard}
          - precip_unit_scale: float (optional) - multiply precip by this factor

    Returns
    -------
    rainfall : xr.DataArray
        Rainfall data with standardized dimensions ['lat', 'lon', 'time'].

    Raises
    ------
    FileNotFoundError
        If no file matching the pattern is found for the given year.
    """
    data_folder = cfg["data_folder"]
    file_pattern = cfg.get("file_pattern", "data_{year}.nc")
    precip_var = cfg.get("precip_var", "RAINFALL")
    dim_mapping = cfg.get("dim_mapping", {})
    unit_scale = cfg.get("precip_unit_scale", None)

    # Resolve filename
    filename = file_pattern.format(year=year)
    filepath = os.path.join(data_folder, filename)

    if not os.path.exists(filepath):
        # Try alternative pattern as fallback
        alt_filename = f"{year}.nc"
        alt_filepath = os.path.join(data_folder, alt_filename)
        if os.path.exists(alt_filepath):
            filepath = alt_filepath
        else:
            available = [f for f in os.listdir(data_folder) if f.endswith(".nc")]
            raise FileNotFoundError(
                f"No file found for year {year} in '{data_folder}'.\n"
                f"Tried: '{filename}', '{alt_filename}'.\n"
                f"Available files: {available}"
            )

    print(f"Loading {year} rainfall from: {filepath}")
    ds = xr.open_dataset(filepath)

    # Extract precipitation variable
    if precip_var not in ds:
        available_vars = list(ds.data_vars)
        raise KeyError(
            f"Variable '{precip_var}' not found in {filepath}. "
            f"Available variables: {available_vars}"
        )

    rainfall = ds[precip_var]

    # Rename dimensions to standard names
    rename = {k: v for k, v in dim_mapping.items() if k in rainfall.dims}
    if rename:
        rainfall = rainfall.rename(rename)
        print(f"Renamed dimensions: {rename}")

    # Apply unit conversion if specified
    if unit_scale is not None:
        rainfall = rainfall * unit_scale
        print(f"Applied unit scale factor: {unit_scale}")

    return rainfall
