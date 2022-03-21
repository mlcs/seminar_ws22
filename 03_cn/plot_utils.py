# Plotting
# ========
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp
import cartopy.crs as ccrs


def create_map(ax=None, projection='PlateCarree',
               central_longitude=0):
    """Generate cartopy figure for plotting.

    Args:
        ax ([type], optional): [description]. Defaults to None.
        ctp_projection (str, optional): [description]. Defaults to 'EqualEarth'.
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
             ax=None, cmap='RdBu_r',
             bar=True,
             projection='PlateCarree',
             label=None,
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


def create_map_for_da(da, data, name='var'):
    if 'time' in da.dims:
        ds = da.mean(dim='time')
    else:
        ds = da
    xr_ds = xr.DataArray(
        data=data,
        dims=ds.dims,
        coords=ds.coords,
        name=name,
    )
    return xr_ds


def plot_edges(
    cnx,
    edges,
    ax=None,
    central_longitude=0,
    projection="EqualEarth",
    plot_points=False,
    **kwargs,
):

    ax = create_map(
        ax=ax, projection=projection,
        central_longitude=central_longitude
    )

    lw = kwargs.pop("lw", 1)
    alpha = kwargs.pop("alpha", 1)
    c = kwargs.pop("color", "k")

    for i, (u, v) in enumerate(edges):
        lon_u = cnx.nodes[u]["lon"]
        lat_u = cnx.nodes[u]["lat"]
        lon_v = cnx.nodes[v]["lon"]
        lat_v = cnx.nodes[v]["lat"]
        if plot_points is True:
            ax.scatter(
                [lon_u, lon_v], [lat_u, lat_v], c="k",
                transform=ccrs.PlateCarree(), s=1
            )

        ax.plot(
            [lon_u, lon_v],
            [lat_u, lat_v],
            c=c,
            linewidth=lw,
            alpha=alpha,
            transform=ccrs.Geodetic(),
            zorder=-1,
        )  # zorder = -1 to always set at the background

    return {"ax": ax, "projection": projection}
