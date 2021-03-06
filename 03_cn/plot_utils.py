# Plotting
# ========
from scipy.cluster.hierarchy import dendrogram
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy as ctp
import cartopy.crs as ccrs


def create_map(da=None, ax=None, projection='PlateCarree',
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
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = plt.axes(projection=proj)
        # axes properties
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, dms=True,
                          x_inline=False, y_inline=False)
        gl.left_labels = True
        gl.right_labels = False
        ax.add_feature(ctp.feature.BORDERS, linestyle=':')

        if da is not None:
            min_ext_lon = float(np.min(da.coords["lon"]))
            max_ext_lon = float(np.max(da.coords["lon"]))
            min_ext_lat = float(np.min(da.coords["lat"]))
            max_ext_lat = float(np.max(da.coords["lat"]))
            # print(min_ext_lon, max_ext_lon, min_ext_lat, max_ext_lat)
            ax.set_extent(
                [min_ext_lon, max_ext_lon, min_ext_lat, max_ext_lat],
                crs=ccrs.PlateCarree(central_longitude=central_longitude)
            )
        # This is just to avoid a plotting bug in cartopy
        if projection != 'PlateCarree':
            ax.set_global()
    return ax


def plot_map(dmap, central_longitude=0,
             vmin=None, vmax=None,
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
    ax = create_map(da=dmap, ax=ax, projection=projection,
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
    orientation = kwargs.pop('orientation', 'vertical')
    if bar:
        label = dmap.name if label is None else label
        cbar = plt.colorbar(im, extend='both', orientation=orientation,
                            label=label, ax=ax, **kwargs)

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
    projection="PlateCarree",
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


def plot_dendrogram(model, **kwargs):
    # Adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def fancy_dendrogram(*args, **kwargs):
    """
    Inspired from
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

    Returns
    -------
    Dendogramm data as scipy dataset.

    """
    from matplotlib import ticker

    fig, ax = plt.subplots(figsize=(9, 9))
    model = kwargs.pop('model', None)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)
    lw = kwargs.pop('lw', None)
    if lw is None:
        lw = 1
    with plt.rc_context({'lines.linewidth': lw}):
        ddata = dendrogram(linkage_matrix,
                           *args, **kwargs)
    if not kwargs.get('no_plot', False):

        ax.set_xlabel('sample index (cluster size) ')
        ax.set_ylabel('Cluster Level')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                ax.plot(x, y, 'o', c=c)
                ax.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                            textcoords='offset points',
                            va='top', ha='center')
        if max_d:
            ax.axhline(y=max_d, c='k', lw=4, ls='--')
    for axis in [ax.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))

    return ddata
