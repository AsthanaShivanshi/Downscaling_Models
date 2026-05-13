import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import string


import matplotlib.gridspec as gridspec

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

cmap= "viridis"

parser = argparse.ArgumentParser(description="Plot precipitation frames for Switzerland")
parser.add_argument("--date", type=str, default="2019-04-01", help="Date to plot (format: YYYY-MM-DD)")
args = parser.parse_args()
date = args.date

coarse_precip_file = "../GCM_pipeline/EUROCORDEX_11_RCP8.5/pr_Swiss/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc"
bc_precip_file = "../GCM_pipeline/EUROCORDEX_11_RCP8.5_BC/EQM/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc"
bc_bicubic_precip_file = "../GCM_pipeline/EUROCORDEX_11_RCP8.5_BC/EQM/pr_bicubic_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc"
bc_bicubic_unet_precip_file = "../GCM_pipeline/ALP-FINE_8.5/EQM/UNet/UNet_RCP85_2021-2030_tas_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc"
bc_bicubic_ddim_precip_file = "../GCM_pipeline/ALP-FINE_8.5/EQM/DDIM/DDIM_6samples_RCP85_2027-2027_tas_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r1i1p1_rcp85_1971-2099.nc"

def get_latlon(ds, var):
    arr = ds[var]
    if 'lat' in arr.dims and 'lon' in arr.dims:
        lat = arr['lat'].values if 'lat' in arr.coords else ds['lat'].values
        lon = arr['lon'].values if 'lon' in arr.coords else ds['lon'].values
    elif 'latitude' in arr.dims and 'longitude' in arr.dims:
        lat = arr['latitude'].values
        lon = arr['longitude'].values
    else:
        lat = ds['lat'].values
        lon = ds['lon'].values
    return lat, lon

def load_precip_frame_with_coords(file, varnames=["pr", "precip", "RhiresD"], date=None, sample_dim=None):
    ds = xr.open_dataset(file)
    for var in varnames:
        if var in ds.variables:
            arr = ds[var]
            if date is not None and "time" in arr.dims:
                if np.datetime64(date) not in arr["time"].values:
                    arr = arr.sel(time=date, method="nearest")
                else:
                    arr = arr.sel(time=date)
            if sample_dim and sample_dim in arr.dims:
                arr = arr
            else:
                arr = arr.expand_dims(sample=[0])  # Add fake sample dim for consistency
            data = np.clip(arr.values, 0, None)
            data = np.ma.masked_invalid(data)
            lat, lon = get_latlon(ds, var)
            ds.close()
            return data, lat, lon
    ds.close()
    return None, None, None

# Load frames
frames = []
lats = []
lons = []
labels = []
colors = []

# Row 1: Coarse, EQM, EQM+Bicubic
frame, lat, lon = load_precip_frame_with_coords(coarse_precip_file, ["pr"], date)
frames.append(frame[0])
lats.append(lat)
lons.append(lon)
labels.append("Coarse")
colors.append("black")

frame, lat, lon = load_precip_frame_with_coords(bc_precip_file, ["pr"], date)
frames.append(frame[0])
lats.append(lat)
lons.append(lon)
labels.append("EQM")
colors.append("blue")

frame, lat, lon = load_precip_frame_with_coords(bc_bicubic_precip_file, ["pr"], date)
frames.append(frame[0])
lats.append(lat)
lons.append(lon)
labels.append("EQM+Bicubic")
colors.append("purple")

# Row 2: UNet mean (centered)
frame, lat, lon = load_precip_frame_with_coords(bc_bicubic_unet_precip_file, ["pr", "precip"], date, sample_dim="sample")
unet_mean = None
if frame is not None:
    unet_mean = np.mean(frame, axis=0)
frames.append(unet_mean)
lats.append(lat)
lons.append(lon)
labels.append("EQM+Bicubic+UNet Mean")
colors.append("green")

# Rows 3 & 4: 6 DDIM samples
frame, lat, lon = load_precip_frame_with_coords(bc_bicubic_ddim_precip_file, ["pr", "precip"], date, sample_dim="sample")
ddim_samples = []
if frame is not None:
    for i in range(frame.shape[0]):
        ddim_samples.append(frame[i])
        lats.append(lat)
        lons.append(lon)
        labels.append(f"DDIM Sample {i+1}")
        colors.append("red")

frames += ddim_samples



fig = plt.figure(figsize=(18, 16), facecolor='white')
gs = gridspec.GridSpec(3, 6, figure=fig)  # 3 rows, 6 columns

# Top row: 3 plots, each spans 2 columns
axes = [fig.add_subplot(gs[0, i*2:(i+1)*2], projection=ccrs.PlateCarree()) for i in range(3)]

# Middle row: 2 big plots, each spans 3 columns
axes += [
    fig.add_subplot(gs[1, 0:3], projection=ccrs.PlateCarree()),  # UNet Mean (center left)
    fig.add_subplot(gs[1, 3:6], projection=ccrs.PlateCarree()),  # DDIM Sample 1 (center right)
]

# Bottom row: 3 plots, each spans 2 columns
axes += [fig.add_subplot(gs[2, i*2:(i+1)*2], projection=ccrs.PlateCarree()) for i in range(3)]

# Map each plot to its position in the grid
plot_frames = [
    (frames[0], lats[0], lons[0], labels[0], colors[0]),  # Top left
    (frames[1], lats[1], lons[1], labels[1], colors[1]),  # Top center
    (frames[2], lats[2], lons[2], labels[2], colors[2]),  # Top right
    (frames[3], lats[3], lons[3], labels[3], colors[3]),  # UNet Mean (middle left, big)
    (ddim_samples[0], lats[4], lons[4], "DDIM Sample 1", "red"),  # DDIM Sample 1 (middle right, big)
    (ddim_samples[1], lats[4], lons[4], "DDIM Sample 2", "red"),  # Bottom left
    (ddim_samples[2], lats[4], lons[4], "DDIM Sample 3", "red"),  # Bottom center
    (ddim_samples[3], lats[4], lons[4], "DDIM Sample 4", "red"),  # Bottom right
]

subplot_labels = list(string.ascii_uppercase)
mesh = None

for idx, pf in enumerate(plot_frames):
    ax = axes[idx]
    frame, lat, lon, label, color = pf
    if frame is not None and lat is not None and lon is not None:
        mesh = ax.pcolormesh(lon, lat, frame, cmap=cmap, vmin=0, vmax=10, shading='auto')
    ax.text(0.02, 0.98, subplot_labels[idx], transform=ax.transAxes, fontsize=28, fontweight='bold',
            va='top', ha='left', color='black', bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round'))
    ax.set_title(label, fontsize=22, color=color, pad=16, fontweight='bold')
    ax.coastlines(resolution='10m', linewidth=2)
    ax.add_feature(cfeature.BORDERS, linewidth=1.5)
    ax.set_extent([5.8, 10.6, 45.7, 47.9])
    ax.axis("off")

if mesh is not None:
    cbar = plt.colorbar(mesh, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04)
    cbar.set_label(f'Precipitation (mm/day) for {date}', fontsize=18)

plt.savefig(f"../GCM_pipeline/EUROCORDEX_11_RCP8.5_BC/outputs/precip_frames_{date}.png", dpi=1000)
plt.show()