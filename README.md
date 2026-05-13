# Monsoon Onset Climatology

A config-driven Python package to compute and visualise climatological monsoon onset dates from any gridded daily rainfall dataset.

---

## Repository Structure

```
icts_mcdm_2026/
├── configs/
│   ├── imd_1deg.yaml          # IMD 1-degree config
│   ├── imerg_1deg.yaml        # IMERG 1-degree config
│   └── era5.yaml              # ERA5 config (example)
├── monsoon_onset/
│   ├── __init__.py
│   ├── loader.py              # Dataset loader
│   ├── detector.py            # Onset detection algorithm
│   ├── climatology.py         # Multi-year + climatological onset
│   └── plotter.py             # Onset map plotting
├── output/                    # Auto-created: NetCDF + figures saved here
├── run_onset.py               # CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-username/icts_mcdm_2026.git
cd icts_mcdm_2026
pip install -r requirements.txt
```

---

## Quick Start

### 1. Edit the config file

Edit `configs/imd_1deg.yaml` to point to your data:

```yaml
dataset:
  data_folder: "/path/to/your/imd_1deg_folder"
  threshold_file: "/path/to/mwset1x1.nc4"
  file_pattern: "data_{year}.nc"
  precip_var: "RAINFALL"
  dim_mapping:
    LATITUDE: "lat"
    LONGITUDE: "lon"
    TIME: "time"

onset:
  wet_spell_window: 5
  dry_spell_window: 10
  dry_spell_threshold: 5
  dry_spell_check_days: 30

years:
  start: 1901
  end: 2025
```

### 2. Run from command line

```bash
# Use defaults from config
python run_onset.py --config configs/imd_1deg.yaml

# Override years
python run_onset.py --config configs/era5.yaml --start_year 1979 --end_year 2024

# Custom output dir
python run_onset.py --config configs/imd_1deg.yaml --output_dir /path/to/results
```

### 3. Use as a Python library

```python
import yaml
import xarray as xr
from monsoon_onset import (
    compute_multi_year_onset,
    compute_climatological_onset,
    plot_climatological_onset,
)

# Load config
with open("configs/imd_1deg.yaml") as f:
    cfg = yaml.safe_load(f)

# Load threshold
thresh_da = xr.open_dataset(cfg["dataset"]["threshold_file"])["MWmean"]

# Compute onset for all years
onset_ds = compute_multi_year_onset(
    dataset_cfg=cfg["dataset"],
    onset_cfg=cfg["onset"],
    years_cfg=cfg["years"],
    threshold_da=thresh_da,
)

# Compute climatology
clim_doy, clim_date, valid_count = compute_climatological_onset(onset_ds)

# Plot
plot_climatological_onset(clim_doy, colormap="Blues", save_path="onset_clim.pdf")
```

---

## Adding a New Dataset

1. Copy an existing config, e.g. `cp configs/era5.yaml configs/my_dataset.yaml`
2. Update the following fields:
   - `data_folder` — path to your NetCDF files
   - `file_pattern` — filename pattern using `{year}` as placeholder
   - `precip_var` — name of the precipitation variable
   - `dim_mapping` — map non-standard dimension names to `lat`, `lon`, `time`
   - `precip_unit_scale` — optional unit conversion factor
3. Run: `python run_onset.py --config configs/my_dataset.yaml`

---

## Onset Algorithm

For each grid point, onset is defined as:

> The **first wet spell** on or after `start_month/start_day` where:
> 1. Rainfall on day 1 > `min_first_day_rain` mm, AND
> 2. The `wet_spell_window`-day total > threshold
>
> AND the wet spell is **NOT followed by a dry spell** (any `dry_spell_window`-day period  
> with total < `dry_spell_threshold` mm) within the next `dry_spell_check_days` days.

All parameters are configurable per dataset via the YAML config.

---

## Config Reference

| Key | Description | Default |
|-----|-------------|---------|
| `dataset.data_folder` | Directory containing NetCDF files | — |
| `dataset.file_pattern` | Filename pattern with `{year}` | `data_{year}.nc` |
| `dataset.precip_var` | Precipitation variable name | `RAINFALL` |
| `dataset.dim_mapping` | Dict of dimension renames | `{}` |
| `dataset.precip_unit_scale` | Multiply precip by this factor | `null` |
| `dataset.threshold_file` | Path to threshold NetCDF | — |
| `dataset.threshold_var` | Threshold variable name | `MWmean` |
| `onset.start_month` | Month to start onset search | `4` |
| `onset.start_day` | Day to start onset search | `1` |
| `onset.wet_spell_window` | Wet spell window (days) | `5` |
| `onset.dry_spell_window` | Dry spell window (days) | `10` |
| `onset.dry_spell_threshold` | Dry spell total threshold (mm) | `5` |
| `onset.dry_spell_check_days` | Days after onset to check for dry spell | `30` |
| `onset.min_first_day_rain` | Min rain on first day of wet spell (mm) | `1` |
| `onset.mok` | Apply MOK June-2 filter | `false` |
| `years.start` | First year to process | `1901` |
| `years.end` | Last year to process | `2024` |
| `output.output_dir` | Output directory | `./output` |
| `output.save_nc` | Save NetCDF output | `true` |
| `output.filename_prefix` | Prefix for output filenames | `onset` |
