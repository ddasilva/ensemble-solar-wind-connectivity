from datetime import timedelta
import glob
import os

from astropy.io import fits
from astropy.time import Time
from cartopy import crs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from joblib import Parallel, delayed
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib import ticker
import numpy as np
import pandas as pd
from PIL import Image
import pylab as plt
import scipy.ndimage
from sklearn.neighbors import KernelDensity


# Julian dates stored in text files must have this bias added to them when they
# are converted to conventional julian dates (eg, with AstroPy).
JULDATE_BIAS = 2440000.0

# Length of the sliding window that holds footpoints contributing to each
# distribution.
SLIDING_WINDOW_LENGTH = timedelta(days=1)

# Minimum number of points in a window required to generate a footpoint
# distribution.
SLIDING_WINDOW_MIN_PTS = 25

# Scale Time (H in exp(-DeltaT/H)) for exponential decay weighting, where-in
# footpoints are weighted by their time offset from center of window.
EXP_DECAY_SCALE_TIME = timedelta(hours=3)

# Bandwidth (standard deviation) parameter of Kernel Density Estimation, in
# radians
KDE_BANDWIDTH = np.deg2rad(0.31)

# Nominal solar wind, used for describing the time the parcel departs
NOMINAL_SOLAR_WIND_DELAY = timedelta(days=4)

# Name of base colormap used for distribution visualization. The actual colormap
# is this but with the lower end made gradually more transparent (see the
# function get_footpoint_cmap().
BASE_FOOTPOINT_COLORMAP = 'autumn_r'

# Color that coronal holes are plotted in when background='mixed' or
# background='coronal_holes' (see the function get_average_ch_cmap()).
CORONAL_HOLE_COLOR = [50, 205, 50] # limegreen

# Minimum Enclosed Probability in for a label to be attached to the polygon.
MIN_ENCLOSED_PROB_FOR_LABEL = 0.01

# Default number of days ahead, which controls which field line files
# used to drive the collection of points for the kernel density estimation
DEFAULT_DAYS_AHEAD = 4


def estimate_footpoint_distribution(prediction_time, df_window, nlat=360, nlon=720):
    """Calculate a footpoint probability distribution function (PDF) from
    ensemble predictions.

    Uses a Kernel Density Estimate (KDE) model over the sphere to sum gaussian
    distributions centered as the ensemble predictions.

    Args
      prediction_time: Nominal time of the window
      df_window: Pandas dataframe holding footpoints in window
      nlat: Number of latitude points in the gridded footpoint PDF
      nlon: Number of longitude points in the gridded footpoint PDF
    Returns
      prob_map: Footpoint probability density map normalized to peak
      prob_map_lon, prob_map_lat: Reference grid for the probability map
    """
    # Fit KDE with all data
    # -----------------------------------------------------------------------
    kde_points = pd.DataFrame({
        'lat': np.deg2rad(df_window.lat_footpoint),
        'lon': np.deg2rad(df_window.lon_footpoint)
    }).values

    delta_t = np.abs(df_window.juldate - prediction_time).dt.total_seconds()
    scale = EXP_DECAY_SCALE_TIME.total_seconds()
    weights = np.exp(-delta_t/scale)

    kde = KernelDensity(metric='haversine', kernel='gaussian',
                        bandwidth=KDE_BANDWIDTH)
    kde.fit(kde_points, None, weights)

    # Use KDE probability distribution to predict PDF at each point on sphere
    # -----------------------------------------------------------------------
    lat_vec = np.linspace(-np.pi / 2, np.pi / 2, nlat)
    lon_vec = np.linspace(-np.pi, np.pi, nlon)
    prob_map_lat, prob_map_lon = np.meshgrid(lat_vec, lon_vec)

    map_points = np.vstack([prob_map_lat.ravel(), prob_map_lon.ravel()]).T
    prob_map = (
        np.exp(kde.score_samples(map_points))
        .reshape(prob_map_lon.shape)
    )
    prob_map /= prob_map.sum()
    prob_map /= prob_map.max()

    return prob_map, prob_map_lon, prob_map_lat


def plot_footpoint_distribution(
        df_window, prob_map, prob_map_lon, prob_map_lat, prediction_time,
        satellite_name, wsa_out_dir):
    """Plot a footpoint probability distribution function (PDF) from
    ensemble predictions.

    Args
      df_window: Pandas dataframe holding footpoints in window
      prob_map: Footpoint probability density map normalized to peak
      prob_map_lon, prob_map_lat: Reference grid for the probability map
      prediction_time: Nominal time of the window
      satellite_name: Name of satellite
      wsa_out_dir: directory to find WSA fits files in
    See Also:
      estimate_footpoint_distribution(): For prob_map* arguments
    """
    # Plot background
    # -----------------------------------------------------------------------
    _, earth_longitude = get_subsat_loc(
        prediction_time, glob_match=f'data/Earth_locs.dat')

    delay = NOMINAL_SOLAR_WIND_DELAY
    depart_time = prediction_time - delay
    central_longitude = earth_longitude

    projection = crs.PlateCarree(central_longitude=central_longitude)
    fig = plt.figure(figsize=(25, 18))

    # left bottom width height
    ax = fig.add_axes([.07, .05, .88, .90], projection=projection)
    ax.set_global()

    # Show magnetogram
    nominal_row_idx = (
        df_window
        .juldate
        .searchsorted(prediction_time)
    )
    wsa_file = os.path.join(wsa_out_dir, df_window.output_file.iloc[nominal_row_idx])
    wsa_fits = fits.open(wsa_file)
    carrlong = wsa_fits[0].header['CARRLONG']
    wsa_fits.close()

    wsa_coordsys_transform = crs.PlateCarree(
        central_longitude=(carrlong % 360) - 180
    )

    grey_image = Image.new('RGB', (360, 180)) # width, height
    grey_pix = grey_image.load()
    tmp_fits = fits.open(wsa_file)
    photosphere_from_wsa_fits(grey_pix, tmp_fits, 2.0)
    tmp_fits.close()

    ax.imshow(
        grey_image, origin='upper', transform=wsa_coordsys_transform,
        extent=[-180, 180, -90, 90]
    )

    subsat_label = 'Sub-sat Longitude at Solar Wind Forecast'
    ax.plot([earth_longitude, earth_longitude],
            [-90, 90], label=subsat_label,
            color='red', linestyle='dashed',
            transform=crs.PlateCarree())

    # Show coronal hole map
    avg_ch_map = get_average_ch_map(df_window, wsa_out_dir)
    ax.imshow(
        avg_ch_map[::-1, :], origin='upper', cmap=get_ch_cmap(),
        vmin=0, vmax=1, transform=wsa_coordsys_transform,
        extent=[-180, 180, -90, 90]
    )

    # Plot Footpoint PDF as a filled contour plot
    # -----------------------------------------------------------------------
    vmin, vmax = 0, 1
    ticks = np.linspace(vmin, vmax, 25)

    cs = ax.contourf(
        np.rad2deg(prob_map_lon), np.rad2deg(prob_map_lat), prob_map,
        levels=ticks, vmin=vmin, vmax=vmax, cmap=get_footpoint_cmap(),
        zorder=999., transform=crs.PlateCarree(),
    )

    cax, kw = mpl.colorbar.make_axes(
        ax, location='bottom', pad=.07, fraction=.08, shrink=.8,
        anchor='S',
        panchor=(0.5, 0.40)
    )
    cbar = plt.colorbar(cs, cax=cax, ticks=ticks, **kw)
    cbar.ax.set_xticklabels([''] * ticks.size)
    cbar.set_label('Relative Probability Density', fontsize=26, color='black')

    # Add enclosing area as contour lines with labels
    # -----------------------------------------------------------------------
    cs = ax.contour(
        np.rad2deg(prob_map_lon), np.rad2deg(prob_map_lat), prob_map,
        cmap='cool', zorder=1000, levels=[1e-7], transform=crs.PlateCarree(),
    )
    points = list(zip(np.rad2deg(prob_map_lon.flatten()),
                      np.rad2deg(prob_map_lat.flatten())))
    paths = cs.collections[0].get_paths()
    tmp = (prob_map / prob_map.sum()).flatten()

    for path in paths:
        mask = path.contains_points(points)
        total_prob_dens = tmp[mask].sum()

        x, y = path.vertices.mean(axis=0)

        if total_prob_dens > MIN_ENCLOSED_PROB_FOR_LABEL:
            ax.text(
                x, y, '%d%%' % round(100*total_prob_dens), color='yellow',
                fontweight='bold', fontsize=22, zorder=1001,
                transform=crs.PlateCarree(),
            )

    # Adjust axes, legend, and title
    # -----------------------------------------------------------------------
    @ticker.FuncFormatter
    def xaxis_major_formatter(x, pos):
        return f'${(x + central_longitude + 360) % 360:.1f}' + r'^{\circ}$'

    ax.set_xticks([0, 60, 120, 180, 240, 300, 360], crs=crs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=crs.PlateCarree())
    ax.tick_params(axis='x', labelsize=26, color='black', labelcolor='black')
    ax.tick_params(axis='y', labelsize=26, color='black', labelcolor='black')
    ax.xaxis.set_major_formatter(xaxis_major_formatter)
    ax.yaxis.set_major_formatter(
        LatitudeFormatter(number_format='.1f')
    )
    ax.set_xlabel('Carrington Longitude', fontsize=26, color='black')

    # Set title
    time_fmt = "%Y-%m-%dT%H:%M"
    title_tmp = 'Solar~Wind'

    ax.set_title(r"$\bf{" + title_tmp + "~Sources~for~" + satellite_name + '}}$'
                 f' (N = {len(df_window.index)})\n'
                 f'Forecast at {prediction_time.strftime(time_fmt)} - '
                 f'Departing around {depart_time.strftime(time_fmt)}\n',
                 fontsize=30, color='black')

    # Setup legend
    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.21),
                    ncol=1, fontsize=26, frameon=False)

    for text in leg.get_texts():
        text.set_color('black')

    # Stonyhurst axis
    stony_ax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
    stony_ax.set_xlabel('Stonyhurst Longitude', fontsize=26, color='black')
    stony_ax.tick_params(axis='x', labelsize=26, color='black', labelcolor='black')
    stony_ax.xaxis.set_major_formatter(
       LongitudeFormatter(dateline_direction_label=False, number_format='.1f')
    )
    stony_ax.yaxis.set_major_formatter(
       LatitudeFormatter(number_format='.1f')
    )
    

def load_footpoints(pred_dir, obs, ahead_days=DEFAULT_DAYS_AHEAD):
    """Read foodpoints from multiple field line files on disk.

    Args
      pred_dir: Path to directory holding field line files
      obs: name of observatory (either AGONG or AHMI)
      ahead_days: Use predictions for +D days ahead of AGONG observation
    Returns
      df_fp: Pandas data frame holding all field line file contents,
        with additional column for file name
    """
    # Search for field line files matching pattern
    glob_pattern = pred_dir + f'/**/*{obs}field_line{ahead_days}*.dat*'
    footpoint_files = glob.glob(glob_pattern, recursive=True)
    footpoint_files.sort()

    if not footpoint_files:
        raise RuntimeError(f'No footpoint files found matching {glob_pattern}')        

    # Print list of files found
    print(f'Found {len(footpoint_files)} files containing footpoints:')

    for footpoint_file in footpoint_files:
        print('   ', os.path.basename(footpoint_file))

    print()

    # Load files and merge into single data frame
    print('Loading footpoint files...')

    dfs = {fname: read_pred_file(fname) for fname in footpoint_files}
    df_fp_rows = []

    for key, df in dfs.items():
        for _, row in df.iterrows():
            df_fp_rows.append([key] + row.tolist())

    df_fp = pd.DataFrame(df_fp_rows, columns=(['file']+df.columns.tolist()))
    df_fp = df_fp.sort_values('juldate').reset_index(drop=True)

    print('Done!')
    
    return df_fp


def get_subsat_loc(time, glob_match=None, _cache={}):
    """Get subsat location at a given time.

    Args
      time: datetime instance (with timezone)
      glob_match: file name to search for using glob (eg '**/Earth_locs.dat')
        this pattern is prefixed with the run dir
    Returns
      subsat_lat, subsat_lon
    """
    pattern = glob_match
    sat_locs_file, *_ = glob.glob(pattern, recursive=True)

    if sat_locs_file in _cache:
        df_locs = _cache[sat_locs_file]
    else:
        df_locs = pd.read_csv(
            sat_locs_file, delim_whitespace=True,
            names=['sat_time_jd', 'sat_lon', 'carrot_num', 'sat_lat', 'radii'],
        )
        _cache[sat_locs_file] = df_locs

    time_jd = Time(time).jd

    sat_lat = np.interp(time_jd, df_locs.sat_time_jd, df_locs.sat_lat)

    # Manually handle interpolation of longitude to account for wrapping
    idx = df_locs.sat_time_jd.searchsorted(time_jd)
    time0, lon0 = (df_locs.sat_time_jd[idx - 1], df_locs.sat_lon[idx - 1])
    time1, lon1 = (df_locs.sat_time_jd[idx], df_locs.sat_lon[idx])

    if lon0 < lon1:
        lon0 += 360    # handle wrapping

    fade_param = (time_jd - time0) / (time1 - time0) # between 0 and 1
    sat_lon = lon0 + fade_param * (lon1 - lon0)
    sat_lon = sat_lon % 360

    return (sat_lat, sat_lon)


def get_ch_cmap(n=10):
    """Gets the colormap used for coronal holes overlay.

    Returns
       custom colormap as a matplotlib object
    """
    custom_cmap = np.zeros((n, 4))
    custom_cmap[:, :-1] = np.array(CORONAL_HOLE_COLOR) / 255
    custom_cmap[:, -1] = np.linspace(0, 0.65, n)
    custom_cmap = ListedColormap(custom_cmap)

    return custom_cmap


def get_footpoint_cmap(split=8, start=0, stop=3):
    """Gets the colormap used for the footpoint distribution.

    Returns
       custom colormap as a matplotlib object
    """
    base_cmap = plt.get_cmap(BASE_FOOTPOINT_COLORMAP)

    j = base_cmap.N // split

    custom_cmap = base_cmap(np.arange(base_cmap.N))
    custom_cmap[:j, -1] = np.exp(-np.linspace(start, stop, j)[::-1])
    custom_cmap = ListedColormap(custom_cmap)

    return custom_cmap


def get_average_ch_map(df_window, wsa_out_dir):
    """Calculate average coronal hole map with values in [0, 1].

    Args
      df_window: Pandas dataframe holding footpoints in window
      wsa_out_dir: directory to find WSA fits files in
    Returns
      ch_map: averaged coronal hole map
    """
    ch_map = np.zeros((90, 180), dtype=float)

    for file in df_window.output_file:
        file_path = os.path.join(wsa_out_dir, file)
        ch_map += fits.getdata(file_path, 0)[6].astype(bool).astype(int)

    ch_map /= len(df_window.index)

    return ch_map


def photosphere_from_wsa_fits(grey_pix, wsa_fits, grid):
    """
    Taken from old plot_chs script translated from IDL (no longer in plot_chs).

    populate a (180, 360) pixel array with photosphere from WSA output file.
    return numpy array of raw intensity values
    """
    phintscaled = bytscl(wsa_fits[0].data[4], ceil=10.0, floor=-10.0)
    if grid > 1.0:
        phintscaled = scipy.ndimage.zoom(phintscaled, grid, order=0)
    phintscaled = scipy.ndimage.filters.uniform_filter(phintscaled)
    phint = wsa_fits[0].data[4]

    # assign greyscale ints to pixels
    nrows = phintscaled.shape[0]

    for row in range(0, 180):
        for col in range(0, 360):
            grey_pixel = (phintscaled[nrows - 1 - row][col],
                          phintscaled[nrows - 1 - row][col],
                          phintscaled[nrows - 1 - row][col])
            grey_pix[col, row] = grey_pixel

    return phint


def read_pred_file(pred_file, no_date_parse=False):
    """Return parsed pred file.

    Args
       pred_file: Path to pred file
       no_date_parse: (optional) set to true to disable parsing dates,
         which is much faster.
    Returns
       Pandas data frame with the juldate column replaced with datetime
       instances.
    """
    kwargs = dict(
        sep='\\s+',
        comment='#',
    )

    if not no_date_parse:
        kwargs.update(dict(
            parse_dates=['juldate'],
            date_parser=lambda strings: [
                Time(float(s) + JULDATE_BIAS, format='jd').to_datetime()
                for s in strings
            ]
        ))

    return pd.read_csv(pred_file, **kwargs)

def bytscl(array, ceil=None, floor=None, top=255, bottom=0):
    """Scale a given array of values to integers in a new range.

    Args
       array: input array of values
       ceil: numeric value of ceiling to enforce for input array
       floor: numeric value of floor to enforce for input array
       top: numeric value of top of new range
       bottom: numeric value of bottom of new range
    Returns
       scaled array of values
    """
    if ceil is None:
        ceil = np.nanmax(array)
    if floor is None:
        floor = np.nanmin(array)
    scaled_array = ((top-bottom+0.9999)*(array-floor)/(ceil-floor)
                    ).astype(np.int16) + bottom

    # Enforce ceiling of new range
    scaled_array = np.minimum(scaled_array, top)

    # Enforce floor of new range
    scaled_array = np.maximum(scaled_array, 0)

    return scaled_array
