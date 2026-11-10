"""
ts_tools.py  --  our toolbox for data cubes and time series
==========================================================
Geoprocesamiento · Maestría en Geomática · Universidad Nacional de Colombia

This file is a Python MODULE: a plain text file that contains functions.
A notebook can use them after writing

    import ts_tools as tt

and then calling, for example, ``tt.compute_ndvi(red, nir)``.

Why keep functions in a .py file instead of inside the notebook?
* The notebook stays short and readable: it tells the story (load -> analyse -> show).
* The same function can be reused in many notebooks (and in your final project).
* If you fix a bug in the .py file, every notebook that imports it gets the fix.

How this file is organised (search for the lines made of "=" signs):
    PART 1  Catalogue and loading           (needs internet: Planetary Computer)
    PART 2  Reflectance, NDVI, mask, duplicates  (the decisions of cube E3)
    PART 3  Writing and reading the cube in HDF5
    PART 4  Preparing time series: table, dips, regularise, fill, smooth

Every function has a "docstring" (the text between triple quotes right below
the "def" line). It explains what the function does, each argument and what it
returns. In the notebook you can read it with  help(tt.function_name).
"""

# -----------------------------------------------------------------------------
# Libraries used everywhere in this module.
# The specialised libraries (pystac_client, odc.stac, h5py, scipy) are imported
# INSIDE the functions that need them. That way, `import ts_tools` works even on
# a computer where, for example, h5py is not installed yet: you only get an error
# if you call a function that really needs the missing library.
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Some constants (fixed values) used by several functions. Writing them once,
# here, avoids typing the same number in many places.
OFFSET_DATE = pd.Timestamp("2022-01-25")   # first Sentinel-2 L2A products with processing baseline 04.00
SCL_CLEAR = (4, 5, 6)                      # SCL classes kept as clear: vegetation, not vegetated, water
EPOCH = pd.Timestamp("1970-01-01")         # origin of the time axis stored in the file (CF convention)


# =============================================================================
# PART 1 · CATALOGUE AND LOADING  (internet needed)
# =============================================================================

def search_sentinel2(bbox, period, max_cloud=80):
    """Search Sentinel-2 Level-2A images on Microsoft Planetary Computer.

    Parameters
    ----------
    bbox : list of 4 numbers
        The area of interest as [west, south, east, north] in degrees
        (longitude/latitude, WGS 84). This is the order STAC expects.
    period : str
        The dates to search, written as "YYYY-MM-DD/YYYY-MM-DD" (start/end).
    max_cloud : number, optional (default 80)
        Keep only images whose WHOLE tile has less than this percentage of
        clouds. It is a coarse filter: the real cleaning is done later, pixel
        by pixel, with the SCL band.

    Returns
    -------
    list
        A list of STAC items (one item = one Sentinel-2 tile on one date).
        Their links are already "signed", so the files can be read.
    """
    import pystac_client
    import planetary_computer

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,   # adds a temporary access token to every link
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=period,
        query={"eo:cloud_cover": {"lt": max_cloud}},   # "lt" means "less than"
    )
    return list(search.item_collection())


def items_table(items):
    """Make a table (pandas DataFrame) with the metadata of each STAC item.

    Parameters
    ----------
    items : list
        The STAC items returned by ``search_sentinel2``.

    Returns
    -------
    pandas.DataFrame
        One row per item, sorted by time, with the columns:
        id, time (UTC, without time-zone label), date (the day only),
        tile (MGRS code), baseline (processing baseline, as a number) and
        cloud (cloud cover of the whole tile, %).
    """
    rows = []
    for it in items:
        rows.append({
            "id": it.id,
            "time": pd.Timestamp(it.datetime).tz_convert(None),     # UTC, label removed
            "tile": it.properties.get("s2:mgrs_tile"),
            "baseline": it.properties.get("s2:processing_baseline"),
            "cloud": it.properties.get("eo:cloud_cover"),
        })
    table = pd.DataFrame(rows)
    table["baseline"] = pd.to_numeric(table["baseline"], errors="coerce")   # "04.00" -> 4.0
    table["date"] = table["time"].dt.floor("D")                            # 2022-02-10 15:26 -> 2022-02-10
    table = table.sort_values(["time", "id"]).reset_index(drop=True)
    return table


def load_raw_cube(items, bbox, crs="EPSG:32618", resolution=10, bands=("B04", "B08", "SCL")):
    """Load the pixels of the items as a cube with ONE LAYER PER ITEM (no fusing yet).

    This is the controlled load of Week 8 (Section 2.2): we decide the grid.

    Parameters
    ----------
    items : list
        STAC items from ``search_sentinel2``.
    bbox : list of 4 numbers
        Area of interest [west, south, east, north] in degrees.
    crs : str, optional (default "EPSG:32618", UTM zone 18 North)
        Coordinate reference system of the cube grid. It must be in metres so
        that `resolution` can be given in metres.
    resolution : number, optional (default 10)
        Pixel size in metres. 10 m is the native size of B04 and B08.
    bands : tuple of str, optional (default red, NIR and SCL)
        The bands (assets) to read.

    Returns
    -------
    xarray.Dataset
        Variables = the bands, dimensions (time, y, x). Values are the raw
        digital numbers (integers). Already read into memory.
    """
    import odc.stac

    raw = odc.stac.load(
        items,
        bands=list(bands),
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        resampling="nearest",          # same grid as the 10 m bands: no interpolation; SCL is categorical
        groupby="id",                  # one layer per item: we apply our own duplicate rule later
        chunks={"x": 512, "y": 512},   # read in blocks (lazy)...
    )
    return raw.compute()               # ...and now really read the pixels into memory


# =============================================================================
# PART 2 · REFLECTANCE, NDVI, CLOUD MASK AND DUPLICATES  (the decisions of cube E3)
# =============================================================================
# These functions work with plain NumPy arrays. A cube is an array with three
# axes in this order:  (time, y, x)  =  (layers, rows, columns).

def offsets_for_layers(layer_times, table, offset_date=OFFSET_DATE):
    """Find the reflectance offset (0 or -0.1) of every layer of the cube.

    Since processing baseline 04.00 (25 January 2022) Sentinel-2 stores
    reflectance = DN / 10000 - 0.1. Older products store DN / 10000.

    Parameters
    ----------
    layer_times : list or pandas.DatetimeIndex
        The acquisition time of each layer of the cube.
    table : pandas.DataFrame
        The table made by ``items_table`` (it contains the baseline of each item).
    offset_date : pandas.Timestamp, optional
        Used ONLY for a layer whose date is not found in the table
        (fallback rule: on or after this date -> offset -0.1).

    Returns
    -------
    numpy.ndarray
        One offset per layer (float32): 0.0 or -0.1.

    Note
    ----
    We look the baseline up by DATE. The two tiles of one date normally share
    the same baseline; if they do not, we keep the highest one.
    """
    baseline_by_date = table.groupby("date")["baseline"].max()   # one baseline per day
    offsets = []
    for t in pd.DatetimeIndex(layer_times):
        day = t.floor("D")
        baseline = baseline_by_date.get(day, np.nan)             # NaN if the day is not in the table
        if np.isnan(baseline):                                   # fallback: use the date
            offsets.append(-0.1 if t >= offset_date else 0.0)
        elif baseline >= 4.0:
            offsets.append(-0.1)
        else:
            offsets.append(0.0)
    return np.array(offsets, dtype="float32")


def to_reflectance(dn, offsets):
    """Convert Sentinel-2 L2A digital numbers into surface reflectance.

    reflectance = DN / 10000 + offset      (offset = 0 or -0.1 for each layer)

    Parameters
    ----------
    dn : numpy array, shape (time, y, x)
        Digital numbers (integers). DN = 0 means "no data".
    offsets : numpy array, shape (time,)
        One offset per layer, from ``offsets_for_layers``.

    Returns
    -------
    numpy array, shape (time, y, x), float32
        Reflectance. Pixels with DN = 0 become NaN ("not a number" = missing).
    """
    dn = dn.astype("float32")
    dn[dn == 0] = np.nan                                   # no data -> NaN BEFORE adding the offset
    # reshape(-1, 1, 1) turns the list of offsets into a "column" with shape (time, 1, 1),
    # so that NumPy adds offset number k to EVERY pixel of layer k ("broadcasting").
    offsets = np.asarray(offsets, dtype="float32").reshape(-1, 1, 1)
    return dn * 1e-4 + offsets


def compute_ndvi(red, nir):
    """Normalised Difference Vegetation Index: (NIR - red) / (NIR + red).

    Parameters
    ----------
    red, nir : numpy arrays of the same shape
        Red (B04) and near-infrared (B08) reflectance.

    Returns
    -------
    numpy array, float32
        NDVI, between -1 and 1. NaN where it cannot be computed (missing data,
        or NIR + red not positive).
    """
    with np.errstate(divide="ignore", invalid="ignore"):   # silence warnings of 0/0 and NaN
        total = nir + red
        ndvi = (nir - red) / total
    ndvi[~(total > 0)] = np.nan                            # undefined where the sum is 0, negative or NaN
    return ndvi.astype("float32")


def clear_mask(scl, classes=SCL_CLEAR):
    """True where the Scene Classification Layer says the pixel is usable.

    Parameters
    ----------
    scl : numpy array
        The SCL band (class codes 0-11).
    classes : tuple of int, optional (default (4, 5, 6))
        The classes we accept as clear: 4 vegetation, 5 not vegetated,
        6 water (a flooded paddy IS water, so we keep it).

    Returns
    -------
    numpy array of True/False, same shape as `scl`.
    """
    return np.isin(scl, classes)


def fuse_same_day(values, times):
    """Merge the layers that belong to the same day into ONE layer.

    Rule: for each pixel, keep the FIRST valid (not NaN) value among the
    layers of that day, in time order. Apply the cloud mask BEFORE this
    function, so that a cloudy copy is skipped and a clear copy is kept.

    Parameters
    ----------
    values : numpy array, shape (time, y, x)
        The layers (for example NDVI), already masked (clouds = NaN).
    times : list or pandas.DatetimeIndex, length = number of layers
        Acquisition time of each layer, in time order.

    Returns
    -------
    fused : numpy array, shape (days, y, x)
        One layer per day.
    days : pandas.DatetimeIndex
        The day of each fused layer.
    n_layers : numpy array, shape (days,)
        How many layers were merged into each day (1 or 2 in Saldaña).
    """
    times = pd.DatetimeIndex(times)
    day_of_layer = times.floor("D")
    days = day_of_layer.unique().sort_values()
    n_days, ny, nx = len(days), values.shape[1], values.shape[2]

    fused = np.full((n_days, ny, nx), np.nan, dtype="float32")   # start with an empty cube
    n_layers = np.zeros(n_days, dtype="int16")
    for k in range(n_days):
        layers_of_this_day = np.where(day_of_layer == days[k])[0]   # positions of the layers of day k
        n_layers[k] = len(layers_of_this_day)
        for i in layers_of_this_day:
            still_empty = np.isnan(fused[k])                     # pixels with no value yet
            fused[k][still_empty] = values[i][still_empty]       # fill them with this layer
    return fused, days, n_layers


def build_ndvi_cube(raw, table, classes=SCL_CLEAR):
    """Apply the decisions of cube E3 and return a clean NDVI cube.

    Steps: (1) sort the layers by time, (2) DN -> reflectance with the offset of
    each layer, (3) NDVI, (4) cloud mask, (5) one layer per day.

    Parameters
    ----------
    raw : xarray.Dataset
        The raw cube from ``load_raw_cube`` (bands B04, B08, SCL).
    table : pandas.DataFrame
        The table from ``items_table``.
    classes : tuple of int, optional (default (4, 5, 6))
        SCL classes accepted as clear.

    Returns
    -------
    dict with the keys:
        "ndvi"       numpy array (days, y, x), float32, NaN = no clear observation
        "time"       pandas.DatetimeIndex, one day per layer
        "x", "y"     numpy arrays with the coordinates of the pixel CENTRES (metres)
        "provenance" pandas.DataFrame, one row per day: n_layers, tiles,
                     baseline, offset, clear_fraction
    """
    times = pd.DatetimeIndex(raw["time"].values)
    order = np.argsort(times.values)                      # positions that put the layers in time order
    times = times[order]

    red_dn = raw["B04"].values[order]
    nir_dn = raw["B08"].values[order]
    scl = raw["SCL"].values[order]

    offsets = offsets_for_layers(times, table)
    red = to_reflectance(red_dn, offsets)
    nir = to_reflectance(nir_dn, offsets)
    ndvi = compute_ndvi(red, nir)
    ndvi[~clear_mask(scl, classes)] = np.nan              # clouds, shadows... -> NaN
    ndvi_day, days, n_layers = fuse_same_day(ndvi, times)

    # Provenance: one row per day, so that every value in the file can be traced back
    prov = []
    for k, day in enumerate(days):
        rows = table[table["date"] == day]
        prov.append({
            "date": day.strftime("%Y-%m-%d"),
            "n_layers": int(n_layers[k]),
            "tiles": ",".join(sorted(rows["tile"].dropna().astype(str).unique())),
            "baseline": float(rows["baseline"].max()) if len(rows) else np.nan,
            "offset": float(offsets[times.floor("D") == day][0]),
            "clear_fraction": float(np.isfinite(ndvi_day[k]).mean()),
        })
    return {"ndvi": ndvi_day, "time": days,
            "x": raw["x"].values.astype("float64"), "y": raw["y"].values.astype("float64"),
            "provenance": pd.DataFrame(prov)}


# =============================================================================
# PART 3 · WRITING AND READING THE CUBE IN HDF5
# =============================================================================

def to_days(times):
    """Convert dates into 'days since 1970-01-01' (the CF convention used in the file).

    Parameters
    ----------
    times : list or pandas.DatetimeIndex
        The dates to convert.

    Returns
    -------
    numpy array of float64: one number per date (fractions = hours of the day).
    """
    return np.asarray((pd.DatetimeIndex(times) - EPOCH) / pd.Timedelta(days=1), dtype="float64")


def from_days(days):
    """Convert 'days since 1970-01-01' back into dates (the opposite of ``to_days``).

    Parameters
    ----------
    days : numbers (list or numpy array)

    Returns
    -------
    pandas.DatetimeIndex
    """
    return pd.DatetimeIndex(EPOCH + pd.to_timedelta(np.asarray(days, dtype="float64"), unit="D")).round("s")


def save_cube_hdf5(path, cube, chunks=(48, 40, 40), attrs=None,
                   title="Sentinel-2 L2A NDVI cube (Week 8 reference cube E3)"):
    """Write the NDVI cube, its coordinates and its provenance to an HDF5 file.

    File layout:
        /                 global attributes (title, CRS, grid, decisions...)
        /x, /y            pixel-centre coordinates (metres)
        /time             days since 1970-01-01 (one value per layer)
        /ndvi             (time, y, x) float32, chunked, compressed, NaN = no data
        /provenance/...   one 1-D dataset per column of the provenance table

    Parameters
    ----------
    path : str
        Name of the file to create, e.g. "saldana_ndvi_cube.h5".
        WARNING: an existing file with this name is overwritten.
    cube : dict
        The dictionary returned by ``build_ndvi_cube``.
    chunks : tuple of 3 int, optional (default (48, 40, 40))
        Chunk shape (time, y, x). The default is designed for READING TIME
        SERIES: one chunk holds about a year of dates for 40 x 40 pixels, so a
        pixel series is read from a single chunk.
    attrs : dict, optional
        Extra global attributes to store (for example the CRS and the rules).
    title : str, optional
        A one-line description of the file, stored as the global attribute "title".

    Returns
    -------
    None. The file is written to disk.
    """
    import h5py

    ndvi = cube["ndvi"].astype("float32")
    n_t, ny, nx = ndvi.shape
    # A chunk cannot be larger than a FIXED dimension (y and x), so we cut it if needed.
    # Time can grow (maxshape None), so its chunk may be longer than today's number of dates.
    chunks = (chunks[0], min(chunks[1], ny), min(chunks[2], nx))

    with h5py.File(path, "w") as f:                       # "w" = create a new file
        # --- global attributes: what this file is and how it was made -------------
        f.attrs["title"] = title
        f.attrs["created"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        for key, value in (attrs or {}).items():
            f.attrs[key] = value

        # --- coordinates -----------------------------------------------------------
        dx = f.create_dataset("x", data=cube["x"])
        dx.attrs["units"] = "m"
        dx.attrs["long_name"] = "x (easting) of pixel centre"
        dy = f.create_dataset("y", data=cube["y"])
        dy.attrs["units"] = "m"
        dy.attrs["long_name"] = "y (northing) of pixel centre"

        dt = f.create_dataset("time", data=to_days(cube["time"]),
                              maxshape=(None,), chunks=(chunks[0],))   # can grow with new dates
        dt.attrs["units"] = "days since 1970-01-01 00:00:00"
        dt.attrs["calendar"] = "standard"

        # --- the data --------------------------------------------------------------
        dn = f.create_dataset(
            "ndvi",
            data=ndvi,
            chunks=chunks,                     # how the cube is cut into blocks on disk
            maxshape=(None, ny, nx),           # the time axis can grow later (new images)
            compression="gzip",                # lossless compression of every chunk...
            compression_opts=4,                # ...with a medium level (1 fast - 9 small)
            shuffle=True,                      # reorder the bytes first: gzip works better
            fillvalue=np.nan,                  # value of parts never written
        )
        dn.attrs["long_name"] = "NDVI, cloud-masked (SCL classes in clear_classes), one layer per day"
        dn.attrs["units"] = "1"
        dn.attrs["valid_range"] = np.array([-1.0, 1.0], dtype="float32")

        # --- provenance: one 1-D dataset per column, aligned with /time --------------
        g = f.create_group("provenance")
        for column in cube["provenance"].columns:
            values = cube["provenance"][column].to_numpy()   # always a plain numpy array
            if values.dtype.kind in ("O", "U"):         # text columns -> bytes (what HDF5 stores)
              # text columns -> bytes (what HDF5 stores)
                values = values.astype("S")
            g.create_dataset(column, data=values, maxshape=(None,), chunks=(chunks[0],))


def describe_hdf5(path):
    """Print the content of an HDF5 file: groups, datasets (shape, type, chunks,
    compression) and the global attributes.

    Parameters
    ----------
    path : str
        The HDF5 file to describe.
    """
    import h5py

    def show(name, obj):                                   # called once for every object in the file
        if isinstance(obj, h5py.Dataset):
            print(f"  {name:28s} shape={str(obj.shape):16s} {str(obj.dtype):8s} "
                  f"chunks={str(obj.chunks):14s} compression={obj.compression}")
        else:
            print(f"{name}/")

    with h5py.File(path, "r") as f:                       # "r" = read only
        print("/")
        f.visititems(show)
        print("\nGlobal attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {str(value)[:100]}")


def load_cube_hdf5(path):
    """Read the whole cube back from an HDF5 file written by ``save_cube_hdf5``.

    Parameters
    ----------
    path : str
        The HDF5 file.

    Returns
    -------
    dict with the same keys as ``build_ndvi_cube``: "ndvi", "time", "x", "y",
    "provenance", plus "attrs" (the global attributes as a dictionary).
    """
    import h5py

    with h5py.File(path, "r") as f:
        cube = {
            "ndvi": f["ndvi"][:],                          # [:] = read everything into memory
            "time": from_days(f["time"][:]),
            "x": f["x"][:],
            "y": f["y"][:],
            "attrs": dict(f.attrs),
        }
        prov = {}
        for column in f["provenance"]:
            values = f["provenance"][column][:]
            if values.dtype.kind == "S":                   # bytes -> normal text
                values = values.astype(str)
            prov[column] = values
        cube["provenance"] = pd.DataFrame(prov)
    return cube


def read_pixel_series(path, row, col):
    """Read the NDVI time series of ONE pixel directly from the HDF5 file.

    Only the chunks that contain this pixel are read and decompressed: this is
    where a good chunk shape makes reading fast.

    Parameters
    ----------
    path : str
        The HDF5 file.
    row, col : int
        Position of the pixel (row = y index from the top, col = x index from the left).

    Returns
    -------
    pandas.Series
        NDVI values indexed by date.
    """
    import h5py

    with h5py.File(path, "r") as f:
        values = f["ndvi"][:, row, col]                    # all dates, one row, one column
        times = from_days(f["time"][:])
    return pd.Series(values, index=times, name=f"pixel ({row}, {col})")


def read_map(path, k):
    """Read ONE date (layer number k) of the NDVI cube from the HDF5 file.

    Parameters
    ----------
    path : str
        The HDF5 file.
    k : int
        Layer number (0 = first date).

    Returns
    -------
    numpy array, shape (y, x)
    """
    import h5py

    with h5py.File(path, "r") as f:
        return f["ndvi"][k, :, :]


def chunks_touched(shape, chunks, start, stop):
    """How many chunks does a read (a "box" of the cube) touch?

    For each axis: number of chunks = (last chunk index) - (first chunk index) + 1.
    The total is the product over the three axes (the cost model of Week 7).

    Parameters
    ----------
    shape : tuple of 3 int
        Shape of the dataset (time, y, x). Only used to check the box.
    chunks : tuple of 3 int
        Chunk shape (time, y, x).
    start : tuple of 3 int
        First index of the box on each axis (included).
    stop : tuple of 3 int
        Last index of the box on each axis (EXCLUDED, as in Python slices).

    Returns
    -------
    int : the number of chunks that must be read entirely.
    """
    total = 1
    for axis in range(3):
        assert 0 <= start[axis] < stop[axis] <= shape[axis], "the box is outside the dataset"
        first_chunk = start[axis] // chunks[axis]
        last_chunk = (stop[axis] - 1) // chunks[axis]
        total = total * (last_chunk - first_chunk + 1)
    return total


# =============================================================================
# PART 4 · PREPARING TIME SERIES
# =============================================================================
# Idea: a cube (time, y, x) can be seen as a TABLE in which each ROW is a date
# and each COLUMN is a pixel. pandas already knows how to resample, interpolate
# and smooth the columns of a table indexed by dates, so we use it.

def cube_to_table(ndvi, times):
    """Turn a cube (time, y, x) into a table: rows = dates, columns = pixels.

    Parameters
    ----------
    ndvi : numpy array, shape (time, y, x)
    times : pandas.DatetimeIndex, one date per layer

    Returns
    -------
    pandas.DataFrame with shape (time, y*x). Pixel (row, col) is the column
    number row * nx + col (see ``pixel_column``).
    """
    n_t, ny, nx = ndvi.shape
    table = pd.DataFrame(ndvi.reshape(n_t, ny * nx), index=pd.DatetimeIndex(times))
    table.index.name = "time"
    return table


def table_to_cube(table, ny, nx):
    """The opposite of ``cube_to_table``: from a table back to a cube.

    Parameters
    ----------
    table : pandas.DataFrame, rows = dates, columns = pixels
    ny, nx : int
        Number of rows (y) and columns (x) of the cube.

    Returns
    -------
    numpy array, shape (dates, ny, nx), float32
    """
    return table.values.reshape(len(table), ny, nx).astype("float32")


def pixel_column(row, col, nx):
    """Column number of pixel (row, col) in the table made by ``cube_to_table``.

    Parameters
    ----------
    row, col : int
        Position of the pixel in the cube.
    nx : int
        Number of columns (x) of the cube.
    """
    return row * nx + col


def remove_dips(table, window=5, threshold=0.15, min_obs=3):
    """Remove sudden DROPS of NDVI that the cloud mask did not catch.

    Residual clouds, haze and shadows LOWER the NDVI of one date. Real crops
    rarely lose more than `threshold` and recover in a few days. A value is
    removed (set to NaN) when it is lower than the median of its neighbours
    minus `threshold`.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows = dates (irregular, as observed), columns = pixels.
    window : int, optional (default 5)
        Number of consecutive dates used to compute the "neighbour median"
        (the date itself + 2 before + 2 after).
    threshold : float, optional (default 0.15)
        How much lower than its neighbours (in NDVI units) a value must be to
        be considered a dip.
    min_obs : int, optional (default 3)
        A value is judged only if at least this many valid values exist in its
        window; otherwise it is kept (not enough information to decide).

    Returns
    -------
    cleaned : pandas.DataFrame, same shape, with the dips set to NaN
    n_removed : int, number of values removed
    """
    neighbours = table.rolling(window, center=True, min_periods=min_obs).median()
    is_dip = table < (neighbours - threshold)           # False where neighbours is NaN
    cleaned = table.mask(is_dip)                         # mask() puts NaN where the condition is True
    return cleaned, int(is_dip.values.sum())


def remove_dips_hampel(table, window=5, k=3.0, min_obs=3, min_sigma=0.02):
    """Remove dips with a Hampel filter: the threshold adapts to the local noise.

    Like remove_dips, each value is compared with the MEDIAN of its neighbours.
    The difference is the threshold: instead of a fixed number (0.15), it is
    k times the local spread of the values, measured with the MAD
    (median absolute deviation). Where the series is calm the threshold is small;
    where it is noisy the threshold is larger.

    Parameters
    ----------
    table : pandas.DataFrame
        Observations, rows = dates, columns = pixels (from cube_to_table).
    window : int, optional (default 5)
        Number of consecutive dates in the window, centred on each date. Must be odd.
    k : float, optional (default 3)
        How many "robust standard deviations" below the median a value must be
        to be removed. Lower k = more values removed.
    min_obs : int, optional (default 3)
        A value is judged only if its window holds at least this many valid values.
    min_sigma : float, optional (default 0.02)
        Smallest spread allowed (in NDVI units). Without it, a window with three
        identical values would have MAD = 0 and any tiny difference would be a "dip".

    Returns
    -------
    cleaned : pandas.DataFrame, same shape, with the dips set to NaN
    n_removed : int, number of values removed
    """
    values = table.values.astype("float64")
    n_dates = values.shape[0]
    half = window // 2

    # 1) local median of each date (exactly as in remove_dips)
    median = table.rolling(window, center=True, min_periods=min_obs).median().values

    # 2) local MAD: for each date, the median distance between the values of its
    #    window and the median of that window
    mad = np.full(values.shape, np.nan)
    for t in range(n_dates):
        first = max(0, t - half)                       # first date of the window
        last = min(n_dates, t + half + 1)               # one after the last date
        block = values[first:last]                     # the dates of the window, all pixels
        distance = np.abs(block - median[t])           # distance of each value to the median
        mad[t] = np.nanmedian(distance, axis=0)        # median distance, pixel by pixel

    # 3) robust standard deviation: 1.4826 x MAD (with a floor, min_sigma)
    sigma = np.maximum(1.4826 * mad, min_sigma)

    # 4) a dip = a value more than k sigmas BELOW the local median (only downwards:
    #    clouds lower NDVI)
    is_dip = values < (median - k * sigma)             # False where median is NaN
    cleaned = table.mask(is_dip)
    return cleaned, int(is_dip.sum())

def regularize(table, step="SMS"):
    """Put the observations on a REGULAR time axis: one value per period.

    Every period (for example every month) receives the MEDIAN of the clear
    observations that fall inside it. Periods with no observation get NaN.

    Parameters
    ----------
    table : pandas.DataFrame
        Rows = dates (irregular), columns = pixels.
    step : str, optional (default "SMS")
        Length of each period, with pandas codes:
        "SMS" = half month (periods starting on the 1st and the 15th),
        "MS" = calendar month (labelled with the first day of the month),
        "16D" = 16 days, "7D" = one week.
        Choose a step SHORTER than the events you want to see: a rice season
        lasts about 4 months, so half-month steps keep its shape; monthly
        steps blur it.

    Returns
    -------
    regular : pandas.DataFrame, one row per period (median NDVI)
    n_obs : pandas.DataFrame, same shape, number of observations used in each period
    """
    groups = table.resample(step)                        # groups the dates by period
    return groups.median(), groups.count()


def fill_gaps(table, max_gap=2):
    """Fill SHORT gaps (periods without data) by linear interpolation in time.

    Gaps longer than `max_gap` periods are NOT filled: inventing several months
    of data would be guessing, not interpolating. A short gap at the very start
    or end of the series (no data on one side) is filled with the nearest
    available value.

    Parameters
    ----------
    table : pandas.DataFrame
        REGULAR series (output of ``regularize``), rows = periods, columns = pixels.
    max_gap : int, optional (default 2)
        Longest run of consecutive missing periods that will be filled
        (2 half-months = about one month).

    Returns
    -------
    filled : pandas.DataFrame, same shape
    n_filled : int, number of values filled
    """
    # 1) interpolate EVERY gap; at the two ends of the series, where there is data
    #    on one side only, pandas copies the nearest value (limit_direction="both")
    filled = table.interpolate(method="time", limit_direction="both")

    # 2) measure the length of each run of missing values, row by row,
    #    for all pixels at once (forward count + backward count - 1)
    missing = table.isna().values
    n_t = missing.shape[0]
    count_forward = np.zeros(missing.shape, dtype=int)
    count_backward = np.zeros(missing.shape, dtype=int)
    for t in range(n_t):
        previous = count_forward[t - 1] if t > 0 else 0
        count_forward[t] = np.where(missing[t], previous + 1, 0)
    for t in range(n_t - 1, -1, -1):
        following = count_backward[t + 1] if t < n_t - 1 else 0
        count_backward[t] = np.where(missing[t], following + 1, 0)
    gap_length = count_forward + count_backward - 1        # length of the run each missing value belongs to

    # 3) undo the filling inside gaps that are too long
    too_long = missing & (gap_length > max_gap)
    filled = filled.mask(too_long)
    n_filled = int((missing & filled.notna().values).sum())
    return filled, n_filled


def smooth_savgol(table, window=5, order=2):
    """Smooth each series with a Savitzky-Golay filter.

    For every date, the filter fits a small polynomial (degree `order`) to the
    `window` values around it and keeps the value of the polynomial. It reduces
    noise while keeping the height and timing of peaks better than a moving
    average.

    Parameters
    ----------
    table : pandas.DataFrame
        REGULAR and gap-filled series, rows = periods, columns = pixels.
    window : int, optional (default 5)
        Number of periods used for each fit. Must be ODD and larger than `order`.
        Larger window = smoother curve, but peaks get flattened.
    order : int, optional (default 2)
        Degree of the polynomial (2 = parabola). Higher order follows the data
        more closely (less smoothing).

    Returns
    -------
    pandas.DataFrame, same shape.
    The filter cannot work across missing values (NaN). For pixels that still
    have holes, each STRETCH of consecutive values without holes is smoothed
    separately, if it is at least `window` periods long; shorter stretches are
    left as they are. Missing values stay missing.
    """
    from scipy.signal import savgol_filter

    values = table.values.astype("float64")
    smoothed = values.copy()
    n_t = values.shape[0]

    # 1) Pixels with no hole at all: smooth all of them at once (fast)
    complete = ~np.isnan(values).any(axis=0)
    if complete.any():
        smoothed[:, complete] = savgol_filter(values[:, complete], window_length=window,
                                              polyorder=order, axis=0)   # axis=0: along time

    # 2) Pixels with holes: find each stretch without holes and smooth it alone
    for j in np.where(~complete)[0]:
        series = values[:, j]
        start = None                                       # first period of the current stretch
        for t in range(n_t + 1):
            has_value = t < n_t and not np.isnan(series[t])
            if has_value and start is None:
                start = t                                  # a stretch begins
            elif not has_value and start is not None:      # a stretch ends at t - 1
                if t - start >= window:
                    smoothed[start:t, j] = savgol_filter(series[start:t], window_length=window,
                                                         polyorder=order)
                start = None
    return pd.DataFrame(smoothed, index=table.index, columns=table.columns)
