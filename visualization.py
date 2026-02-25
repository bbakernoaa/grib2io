import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import hvplot.xarray

def plot_aerosol_static(ds: xr.Dataset, var_name: str):
    """
    Track A: Publication-grade static plot using matplotlib and cartopy.
    Mandatory: projection in axes and transform in plot calls.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot data
    data = ds[var_name]
    data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        x='longitude',
        y='latitude',
        cmap='viridis',
        cbar_kwargs={'label': data.attrs.get('units', '')}
    )

    plt.title(f"Aerosol Data: {data.attrs.get('long_name', var_name)}")
    plt.show()

def plot_aerosol_interactive(ds: xr.Dataset, var_name: str):
    """
    Track B: Interactive exploration using hvplot.
    Mandatory: rasterize=True for large grids.
    """
    return ds[var_name].hvplot.quadmesh(
        x='longitude',
        y='latitude',
        rasterize=True,
        geo=True,
        tiles='EsriImagery',
        cmap='viridis',
        title=f"Interactive Aerosol Data: {var_name}"
    )

if __name__ == "__main__":
    # Example usage
    # ds = xr.open_dataset("path/to/grib2", engine="grib2io")
    # plot_aerosol_static(ds, "aod")
    pass
