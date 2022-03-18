# Plotting
# ========
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp
import cartopy.crs as ccrs

def create_map(ax=None, projection='PlateCarrree',
               central_longitude=0):
    """Generate cartopy figure for plotting.

    Args:
        ax ([type], optional): [description]. Defaults to None.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarrree'.
        central_longitude (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns:
        ax (plt.axes): Matplotplib axes object.
    """
    # set projection
    if projection == 'Mollweide':
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    elif projection == 'EqualEarth':
        proj = ccrs.EqualEarth(central_longitude=central_longitude)
    elif projection == 'Robinson':
        proj = ccrs.Robinson(central_longitude=central_longitude)
    elif projection == 'PlateCarree':
        proj = ccrs.PlateCarree(central_longitude=central_longitude)
    else:
        raise ValueError(
            f'This projection {projection} is not available yet!')

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax = plt.axes(projection=proj)

    # axes properties
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.add_feature(ctp.feature.BORDERS, linestyle=':')

    return ax


def plot_map(dmap, central_longitude=0, vmin=None, vmax=None,
             ax=None, cmap='RdBu_r', bar=True,
             projection='PlateCarree', label=None,
             plot_type='colormesh',
             **kwargs):
    """Simple map plotting using xArray.

    Args:
        dmap ([type]): [description]
        central_longitude (int, optional): [description]. Defaults to 0.
        vmin ([type], optional): [description]. Defaults to None.
        vmax ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.
        fig ([type], optional): [description]. Defaults to None.
        cmap (str, optional): [description]. Defaults to 'RdBu_r'.
        bar (bool, optional): [description]. Defaults to True.
        ctp_projection (str, optional): [description]. Defaults to 'PlateCarree'.
        label ([type], optional): [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # create figure
    ax = create_map(ax=ax, projection=projection,
                    central_longitude=central_longitude)

    # set colormap
    cmap = plt.get_cmap(cmap)

    lon_mesh, lat_mesh = dmap.coords["lon"], dmap.coords["lat"]
    # plot map
    if plot_type == "colormesh":
        im = ax.pcolormesh(
            lon_mesh,
            lat_mesh,
            dmap,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )
    elif plot_type == "contourf":
        levels = kwargs.pop("levels", 8)
        im = ax.contourf(
            lon_mesh,
            lat_mesh,
            dmap,
            levels=levels,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extend="both",
        )

    # set colorbar
    shrink = kwargs.pop('shrink', 0.8)
    if bar:
        label = dmap.name if label is None else label
        cbar = plt.colorbar(im, extend='both', orientation='horizontal',
                            label=label, shrink=shrink, ax=ax, **kwargs)


    return {'ax': ax, "im": im}
