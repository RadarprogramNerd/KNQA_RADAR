import matplotlib.pyplot as plt
import pyart
import numpy as np
import fsspec
from datetime import datetime, timezone

def download_latest_nexrad(site):
    """Download the latest NEXRAD radar data from an S3 bucket."""
    try:
        now = datetime.now(timezone.utc)
        fs = fsspec.filesystem("s3", anon=True)
        nexrad_path = now.strftime(
            f"s3://noaa-nexrad-level2/%Y/%m/%d/{site}/{site}%Y%m%d_%H*"
        )
        files = sorted(fs.glob(nexrad_path))
        for file in files:
            if not file.endswith("_MDM"):
                return file
        return None
    except Exception as e:
        print(f"Error in processing: {e}")
        return None

def plot_radar(file):
    radar = pyart.io.read_nexrad_archive("s3://" + file)
    radar.fields["reflectivity"]["data"][:, -10:] = np.ma.masked
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_transition()
    gatefilter.exclude_masked("reflectivity")
    gatefilter.exclude_outside("reflectivity", 10, 80)
    max_range = np.ceil(radar.range["data"].max())
    if max_range / 1e3 > 250:
        max_range = 250 * 1e3
    grid = pyart.map.grid_from_radars(
        (radar,),
        gatefilters=(gatefilter,),
        grid_shape=(30, 441, 441),
        grid_limits=((0, 10000), (-max_range, max_range), (-max_range, max_range)),
        fields=["reflectivity"],
    )
    gdisplay = pyart.graph.GridMapDisplay(grid)
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(8, 8))
        gdisplay.plot_maxcappi(
            field="reflectivity", cmap="pyart_HomeyerRainbow", add_slogan=True, ax=ax
        )
        plt.savefig('radar.png')
        plt.close()

if __name__ == "__main__":
    site = "KNQA"
    file = download_latest_nexrad(site)
    if file:
        plot_radar(file)
    else:
        print("No radar file found.")
