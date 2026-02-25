
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import grib2io

def plot_aerosol_data(ds: xr.Dataset, var_name: str, title: str = "Aerosol Concentration"):
    """
    Plot aerosol data using matplotlib and cartopy (Track A: Publication).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing aerosol data.
    var_name : str
        Variable name to plot.
    title : str, optional
        Plot title.
    """
    # Use the projection information from the dataset attributes
    # In a real scenario, we'd parse crs_wkt or other attrs
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': proj})

    # Plot data
    # Mandatory: transform= in plot calls
    ds[var_name].plot(ax=ax, transform=proj, x='longitude', y='latitude', cmap='YlOrRd')

    ax.coastlines()
    ax.set_title(title)
    plt.show()

def interactive_aerosol_plot(ds: xr.Dataset, var_name: str):
    """
    Interactive aerosol plot using hvplot (Track B: Exploration).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing aerosol data.
    var_name : str
        Variable name to plot.
    """
    try:
        import hvplot.xarray
        # Mandatory: rasterize=True for large grids
        return ds[var_name].hvplot.quadmesh(x='longitude', y='latitude', rasterize=True, cmap='YlOrRd', geo=True)
    except ImportError:
        print("hvplot not installed")
