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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import xarray as xr


# ── helpers ──────────────────────────────────────────────────────────────────

def doy_to_date_string(doy: float) -> str:
    """Convert day-of-year float to 'mm/dd' string using year 2000 as reference."""
    date = pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(doy) - 1)
    return date.strftime("%m/%d")


def get_india_outline(shapefile_path: str = None):
    """
    Load India boundary from a shapefile, or fall back to a bounding box.

    Parameters
    ----------
    shapefile_path : str, optional
        Path to an India shapefile (.shp).

    Returns
    -------
    list of (lon_coords, lat_coords) tuples
    """
    if shapefile_path:
        try:
            import geopandas as gpd
            gdf = gpd.read_file(shapefile_path)
            boundaries = []
            for geom in gdf.geometry:
                polys = [geom] if hasattr(geom, "exterior") else list(geom.geoms)
                for poly in polys:
                    coords = list(poly.exterior.coords)
                    boundaries.append(
                        ([c[0] for c in coords], [c[1] for c in coords])
                    )
            print(f"Loaded India outline from: {shapefile_path}")
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
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    gs = GridSpec(1, 14, figure=fig,
                  hspace=0.1, wspace=0.3,
                  left=0.08, right=0.92, top=0.85, bottom=0.15)
    ax = fig.add_subplot(gs[0, 0:12])

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

    # Axes ticks / limits
    ax.set_xlim([lon[0] - 2, 100])
    ax.set_ylim([lat[0] - 2, lat[-1]])
    yticks = np.arange(lat[0] - 2, lat[-1] + 3, 8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y)}°N" for y in yticks], fontsize=SMALL)
    xticks = np.arange(lon[0] - 2, lon[-1] + 3, 8)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(x)}°E" for x in xticks], fontsize=SMALL)

    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_linewidth(0.5)

    # Colorbar
    pos = ax.get_position()
    cb_h = pos.height * 0.7
    cax = fig.add_axes([pos.x1 + 0.01,
                         pos.y0 + (pos.height - cb_h) / 2,
                         0.025, cb_h])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical", spacing="proportional")
    tick_pos = boundaries[::2]
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels([doy_to_date_string(t) for t in tick_pos])
    cbar.ax.tick_params(labelsize=SMALL, length=2, width=1)
    cbar.ax.minorticks_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    return fig
