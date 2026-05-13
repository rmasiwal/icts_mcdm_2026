"""
monsoon_onset/plotter.py
------------------------
Plotting utilities for climatological onset maps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Rectangle
import xarray as xr


# ── helpers ──────────────────────────────────────────────────────────────────

def doy_to_date_string(doy: float, year: int = 2001) -> str:
    """Convert day-of-year float to 'mm/dd' string.

    Parameters
    ----------
    doy : float
        Day-of-year (1-based).
    year : int, optional
        Reference year used to interpret the DOY. Defaults to 2001 (a
        non-leap year) so that DOY values from non-leap years like 2025
        map correctly. Pass the actual data year when the year matters.
    """
    date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(doy) - 1)
    return date.strftime("%m/%d")


def get_india_outline(shapefile_path: str = None, dataset_cfg: dict = None):
    """
    Load India boundary from a shapefile, or fall back to a bounding box.

    Parameters
    ----------
    shapefile_path : str, optional
        Direct path to an India shapefile (.shp).
    dataset_cfg : dict, optional
        Dataset config dict (from YAML). If ``shapefile_path`` is not given,
        the key ``dataset_cfg["shapefile"]`` is used as the path.

    Returns
    -------
    list of (lon_coords, lat_coords) tuples
    """
    # Resolve path: explicit arg takes priority, then config dict
    path = shapefile_path or (dataset_cfg.get("shapefile") if dataset_cfg else None)

    if path:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(path)
            boundaries = []
            for geom in gdf.geometry:
                polys = [geom] if hasattr(geom, "exterior") else list(geom.geoms)
                for poly in polys:
                    coords = list(poly.exterior.coords)
                    boundaries.append(
                        ([c[0] for c in coords], [c[1] for c in coords])
                    )
            print(f"Loaded India outline from: {path}")
            return boundaries
        except Exception as e:
            print(f"Could not load shapefile: {e}. Using fallback outline.")

    # Fallback bounding box
    print("Using simplified India bounding box outline.")
    return [([68, 97, 97, 68, 68], [8, 8, 37, 37, 8])]


# ── main plotting function ────────────────────────────────────────────────────

def plot_climatological_onset(
    clim_doy: xr.DataArray,
    forecast_cells=None,
    show_all_cells: bool = True,
    colormap: str = "Blues",
    vmin: float = None,
    vmax: float = None,
    n_levels: int = 14,
    map_lw: float = 1.0,
    shapefile_path: str = None,
    title: str = None,
    save_path: str = None,
    dpi: int = 600,
):
    """
    Plot climatological mean onset date map.

    Parameters
    ----------
    clim_doy       : xr.DataArray  - mean onset DOY [lat, lon]
    forecast_cells : list of (lat, lon) tuples, optional
                     Grid cells to outline in black. If None, no outlines drawn.
    show_all_cells : bool
                     If True, colour all grid cells.
                     If False, colour only cells in forecast_cells.
    colormap       : str   - matplotlib colormap name (default 'Blues')
    vmin / vmax    : float - colour scale limits in DOY (default: auto)
    n_levels       : int   - number of discrete colour levels
    map_lw         : float - line width for country/India outline
    shapefile_path : str   - path to India shapefile (optional)
    title          : str   - figure title (optional)
    save_path      : str   - path to save figure (optional)
    dpi            : int   - figure resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    lon = clim_doy.lon.values
    lat = clim_doy.lat.values
    data = clim_doy.values.copy()

    # Ensure data is [lat, lon]
    if data.shape == (len(lon), len(lat)):
        data = data.T

    # Optionally mask to forecast_cells only
    if not show_all_cells and forecast_cells is not None:
        masked = np.full(data.shape, np.nan)
        for lv, lo in forecast_cells:
            li = np.where(lat == lv)[0]
            loi = np.where(lon == lo)[0]
            if len(li) and len(loi):
                masked[li[0], loi[0]] = data[li[0], loi[0]]
        data = masked

    # Colour scale
    valid = data[~np.isnan(data)]
    _vmin = vmin if vmin is not None else float(np.floor(valid.min())) - 2
    _vmax = vmax if vmax is not None else float(np.ceil(valid.max())) + 6
    boundaries = np.linspace(_vmin, _vmax, n_levels)
    cmap_obj = cm.get_cmap(colormap, len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap_obj.N, clip=True)

    # Grid edges for pcolormesh
    lon_sp = lon[1] - lon[0]
    lat_sp = lat[1] - lat[0]
    lon_edges = np.concatenate([lon - lon_sp / 2, [lon[-1] + lon_sp / 2]])
    lat_edges = np.concatenate([lat - lat_sp / 2, [lat[-1] + lat_sp / 2]])
    LON_e, LAT_e = np.meshgrid(lon_edges, lat_edges)

    masked_data = np.ma.masked_invalid(data)

    # ── Figure layout ────────────────────────────────────────────────────────
    SMALL = 8
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    im = ax.pcolormesh(LON_e, LAT_e, masked_data,
                       cmap=cmap_obj, norm=norm, shading="flat")

    # India outline
    for lons_b, lats_b in get_india_outline(shapefile_path):
        ax.plot(lons_b, lats_b, color="black", linewidth=map_lw)

    # Forecast cell outlines
    if forecast_cells is not None:
        for lv, lo in forecast_cells:
            rect = Rectangle(
                (lo - lon_sp / 2, lv - lat_sp / 2),
                lon_sp, lat_sp,
                linewidth=0.75, edgecolor="black", facecolor="none", zorder=10,
            )
            ax.add_patch(rect)

    # Axes ticks / limits — set to exact outer cell edges so spines sit outside pixels
    ax.set_xlim(lon_edges[0], lon_edges[-1])
    ax.set_ylim(lat_edges[0], lat_edges[-1])
    yticks = np.arange(np.ceil(lat[0] / 8) * 8, lat[-1] + 1, 8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}°N" for y in yticks], fontsize=SMALL)
    xticks = np.arange(np.ceil(lon[0] / 8) * 8, lon[-1] + 1, 8)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}°E" for x in xticks], fontsize=SMALL)
    ax.tick_params(length=3, width=0.6)

    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    # datalim: aspect ratio adjusts the data range, not the axes box size
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(False)

    # Thin spine at outer cell edges
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    # Colorbar — attached to ax so no orphan frame is left behind
    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        pad=0.02, shrink=0.85, spacing="proportional")
    tick_pos = boundaries[::2]
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([doy_to_date_string(t) for t in tick_pos])
    cbar.ax.tick_params(labelsize=SMALL, length=2, width=1)
    cbar.ax.minorticks_off()
    cbar.outline.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return fig
