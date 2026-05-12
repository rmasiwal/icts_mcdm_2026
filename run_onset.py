#!/usr/bin/env python
"""
run_onset.py
-------------
Entry-point script: compute multi-year onset and climatology from a YAML config.

Usage
-----
    python run_onset.py --config configs/imd_2deg.yaml
    python run_onset.py --config configs/era5.yaml --start_year 1979 --end_year 2024
"""

import argparse
import os
import yaml
import numpy as np
import xarray as xr

from monsoon_onset import (
    compute_multi_year_onset,
    compute_climatological_onset,
    plot_climatological_onset,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute climatological monsoon onset from a YAML config."
    )
    parser.add_argument("--config", required=True,
                        help="Path to dataset YAML config file")
    parser.add_argument("--start_year", type=int, default=None,
                        help="Override start year from config")
    parser.add_argument("--end_year", type=int, default=None,
                        help="Override end year from config")
    parser.add_argument("--save_nc", action="store_true", default=None,
                        help="Force save NetCDF output")
    parser.add_argument("--output_dir", default=None,
                        help="Override output directory from config")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    onset_cfg = cfg["onset"]
    years_cfg = cfg["years"].copy()
    output_cfg = cfg.get("output", {})

    # Apply CLI overrides
    if args.start_year is not None:
        years_cfg["start"] = args.start_year
    if args.end_year is not None:
        years_cfg["end"] = args.end_year
    if args.output_dir is not None:
        output_cfg["output_dir"] = args.output_dir

    output_dir = output_cfg.get("output_dir", "./output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Load threshold ────────────────────────────────────────────────────────
    thresh_file = dataset_cfg["threshold_file"]
    thresh_var = dataset_cfg.get("threshold_var", "MWmean")
    print(f"Loading threshold from: {thresh_file}")
    thresh_ds = xr.open_dataset(thresh_file)
    threshold_da = thresh_ds[thresh_var]

    # ── Compute multi-year onset ──────────────────────────────────────────────
    onset_dataset = compute_multi_year_onset(
        dataset_cfg=dataset_cfg,
        onset_cfg=onset_cfg,
        years_cfg=years_cfg,
        threshold_da=threshold_da,
    )

    # ── Save NetCDF ───────────────────────────────────────────────────────────
    if output_cfg.get("save_nc", True):
        prefix = output_cfg.get("filename_prefix", "onset")
        sy = years_cfg["start"]
        ey = years_cfg["end"]
        nc_path = os.path.join(output_dir, f"{prefix}_{sy}_{ey}.nc")
        onset_dataset.to_netcdf(nc_path)
        print(f"Saved multi-year onset dataset: {nc_path}")

    # ── Compute climatology ───────────────────────────────────────────────────
    clim_doy, clim_onset_date, valid_count = compute_climatological_onset(onset_dataset)

    # ── Plot ──────────────────────────────────────────────────────────────────
    prefix = output_cfg.get("filename_prefix", "onset")
    plot_path = os.path.join(output_dir, f"{prefix}_clim_map.pdf")
    plot_climatological_onset(
        clim_doy,
        colormap="Blues",
        title=f"Climatological Mean Monsoon Onset ({years_cfg['start']}–{years_cfg['end']})",
        save_path=plot_path,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
