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
    PART 5  Long records: MODIS 16-day NDVI and monthly Sentinel-2   (Session 2, needs internet)
    PART 6  Climatology, anomalies, decomposition and trends   (Session 2)
    PART 7  Helpers for the Session 2 notebooks: build-once files, pixels, simple figures,
            and a map to check pixels on a high-resolution image (folium)

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
            if values.dtype.kind in ("O", "U"):              # text (any pandas version) -> bytes (what HDF5 stores)
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


# =============================================================================
# PART 5 · LONG RECORDS: MODIS (TERRA + AQUA) AND SENTINEL-2 BY MONTH  (Session 2, internet needed)
# =============================================================================
# Trends and anomalies need MANY years. Sentinel-2 starts in 2015-2017; MODIS
# (on the satellite Terra) has 16-day NDVI composites since February 2000.
# Product MOD13Q1, version 061, 250 m: on Planetary Computer it is the
# collection "modis-13Q1-061".

MODIS_COLLECTION = "modis-13Q1-061"
MODIS_PERIODS = 23            # 16-day composites per year from ONE satellite: they start on day 1, 17, 33, ..., 353
MODIS_PERIODS_BOTH = 46       # Terra + Aqua together: one composite every 8 days (Aqua starts on day 9, 25, ...)
MODIS_FILL = -3000            # value of "no data" in the NDVI band of MOD13Q1 / MYD13Q1
AQUA_OFFSET_DAYS = 8          # Aqua (MYD13Q1) composites start 8 days after Terra's (MOD13Q1)


def modis_period(dates, step_days=None):
    """Number of the composite inside its year.

    MODIS composites restart every year on 1 January. With ONE satellite (16-day
    steps) the periods are 1 ... 23 (days of year 1, 17, ..., 353). With Terra AND
    Aqua (8-day steps, Aqua shifted by 8 days) they are 1 ... 46. The last period of
    the year is shorter, so the "period" is the NUMBER of the composite, not a fixed
    number of days.

    Parameters
    ----------
    dates : pandas.DatetimeIndex
        The start date of each composite.
    step_days : int, optional (16 or 8)
        Days between consecutive composites. Default: guessed from the dates
        (8 if the typical spacing is 12 days or less, else 16).

    Returns
    -------
    numpy array of int, values 1 ... 23 (16-day) or 1 ... 46 (8-day).
    """
    dates = pd.DatetimeIndex(dates)
    if step_days is None:
        spacing = np.median(np.diff(np.unique(dates.values)).astype("timedelta64[D]").astype(int)) if len(dates) > 1 else 16
        step_days = 8 if spacing <= 12 else 16
    return (dates.dayofyear.values - 1) // step_days + 1


def modis_time_axis(first_year, last_year, platform="both"):
    """The complete, regular list of composite start dates between two years.

    Parameters
    ----------
    first_year, last_year : int
    platform : str ("terra", "aqua" or "both")
        "terra": days of year 1, 17, ...; "aqua": 9, 25, ...; "both": every 8 days.

    Returns
    -------
    pandas.DatetimeIndex. Used to give the cube a REGULAR time axis: a composite
    missing from the catalogue (for example Aqua before July 2002) becomes a layer
    of NaN instead of disappearing, so that moving averages and decompositions
    stay aligned with the seasons.
    """
    starts = {"terra": [0], "aqua": [AQUA_OFFSET_DAYS], "both": [0, AQUA_OFFSET_DAYS]}[platform]
    out = []
    for year in range(first_year, last_year + 1):
        jan1 = pd.Timestamp(f"{year}-01-01")
        for doy0 in range(0, 366, 16):
            for off in starts:
                d = jan1 + pd.Timedelta(days=doy0 + off)
                if d.year == year:
                    out.append(d)
    return pd.DatetimeIndex(sorted(out))


def _snap_to_composite(times, offset_days):
    """Date of the composite start (day 1 + offset + 16k of the year) that contains each time."""
    out = []
    for t in pd.DatetimeIndex(times):
        doy0 = t.dayofyear - 1 - offset_days
        if doy0 < 0:                                       # belongs to the last composite of the previous year
            prev = pd.Timestamp(f"{t.year - 1}-01-01")
            last = ((prev.replace(month=12, day=31).dayofyear - 1 - offset_days) // 16) * 16 + offset_days
            out.append(prev + pd.Timedelta(days=last))
        else:
            out.append(pd.Timestamp(f"{t.year}-01-01") + pd.Timedelta(days=(doy0 // 16) * 16 + offset_days))
    return pd.DatetimeIndex(out)


def load_modis_cube(bbox, period, crs="EPSG:32618", resolution=250, platform="both",
                    good_classes=(0, 1)):
    """Search and load MODIS 16-day NDVI (MOD13Q1 Terra / MYD13Q1 Aqua, v061) as an NDVI cube.

    Parameters
    ----------
    bbox : list of 4 numbers
        [west, south, east, north] in degrees (WGS 84).
    period : str
        Time range, e.g. "2001-01-01/2015-12-31".
    crs : str, optional (default "EPSG:32618", UTM 18 N)
        Coordinate system of the output grid. MODIS is distributed on a
        sinusoidal grid; odc-stac reprojects it to this CRS.
    resolution : number, optional (default 250)
        Pixel size of the output grid, in metres (native MODIS resolution).
    platform : str, optional ("terra", "aqua" or "both"; default "both")
        "terra" = MOD13Q1, since February 2000, composites starting on day 1, 17, ...
        "aqua"  = MYD13Q1, since July 2002, composites starting on day 9, 25, ...
        "both"  = the two series interleaved: one composite every 8 days. Twice
        the observations, but from two satellites that do not measure exactly
        alike (Terra passes in the morning, Aqua in the afternoon): the
        provenance keeps the platform of every layer so the difference can be checked.
    good_classes : tuple of int, optional (default (0, 1))
        Classes of the "pixel reliability" band that we keep:
        0 = good, 1 = marginal, 2 = snow/ice, 3 = cloudy. The rest become NaN.

    Returns
    -------
    dict with the same keys as ``build_ndvi_cube`` (Session 1):
        "ndvi" (time, y, x) float32 on a REGULAR time axis (missing composites = NaN layers),
        "time", "x", "y", "provenance" (date, period, platform, clear_fraction).
    """
    import pystac_client
    import planetary_computer
    import odc.stac

    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                        modifier=planetary_computer.sign_inplace)
    wanted = ["terra", "aqua"] if platform == "both" else [platform]
    layers, grid = {}, None
    for plat in wanted:
        items = list(catalog.search(collections=[MODIS_COLLECTION], bbox=bbox, datetime=period,
                                    query={"platform": {"eq": plat}}).items())
        if not items:
            continue
        data = odc.stac.load(items, bands=["250m_16_days_NDVI", "250m_16_days_pixel_reliability"],
                             bbox=bbox, crs=crs, resolution=resolution, groupby="solar_day",
                             chunks={}).sortby("time").compute()
        raw = data["250m_16_days_NDVI"].values.astype("float32")
        rel = data["250m_16_days_pixel_reliability"].values
        ndvi = np.where(raw == MODIS_FILL, np.nan, raw * 1e-4).astype("float32")   # digital numbers -> NDVI
        ndvi[~np.isin(rel, good_classes)] = np.nan                                  # unreliable -> NaN
        starts = _snap_to_composite(data["time"].values, AQUA_OFFSET_DAYS if plat == "aqua" else 0)
        for k, d in enumerate(starts):
            layers[d] = (ndvi[k], plat)
        grid = (data["x"].values.astype("float64"), data["y"].values.astype("float64"), ndvi.shape[1:])
    if not layers:
        raise ValueError("No MODIS items found: check bbox, period and platform.")
    x, y, shape = grid
    first, last = pd.Timestamp(period.split("/")[0]).year, pd.Timestamp(period.split("/")[1]).year
    axis = modis_time_axis(first, last, platform)          # the REGULAR time axis
    cube = np.full((len(axis),) + shape, np.nan, dtype="float32")
    plat_of = []
    for k, d in enumerate(axis):
        if d in layers:
            cube[k] = layers[d][0]; plat_of.append(layers[d][1])
        else:
            plat_of.append("aqua" if (d.dayofyear - 1) % 16 == AQUA_OFFSET_DAYS else "terra")
    prov = pd.DataFrame({"date": axis.strftime("%Y-%m-%d"),
                         "period": modis_period(axis, 8 if platform == "both" else 16),
                         "platform": plat_of,
                         "clear_fraction": np.isfinite(cube).mean(axis=(1, 2))})
    return {"ndvi": cube, "time": axis, "x": x, "y": y, "provenance": prov}


def build_s2_monthly_cube(bbox, years, crs="EPSG:32618", resolution=20, max_cloud=80):
    """Several years of Sentinel-2 NDVI as ONE value per pixel and MONTH (Session 2).

    It repeats the chain of Session 1 year by year (search -> load one layer per
    item -> offset, NDVI, SCL mask, fuse duplicates with ``build_ndvi_cube``) and
    then summarises each month with the MEDIAN of its clear observations. Working
    year by year keeps the memory small.

    Parameters
    ----------
    bbox : list of 4 numbers
        [west, south, east, north] in degrees (WGS 84).
    years : iterable of int
        The years to load, e.g. range(2018, 2026).
    crs : str, optional (default "EPSG:32618")
        Coordinate system of the grid (metres).
    resolution : number, optional (default 20)
        Pixel size in metres. 20 m (instead of the native 10 m) makes the
        download four times lighter and the trend map four times faster; for a
        question about trends over a district it is still 150 times finer than MODIS.
    max_cloud : number, optional (default 80)
        Coarse filter on the cloud cover of the whole tile, as in Session 1.

    Returns
    -------
    dict with "ndvi" (months, y, x), "time" (first day of each month), "x", "y" and
    "provenance" (date, n_obs_median = median number of clear observations behind
    each pixel value, clear_fraction = share of pixels with a value).
    ⏳ Slow: several minutes per year. Build it once and save it with ``save_cube_hdf5``.
    """
    layers, times, n_obs, x, y = [], [], [], None, None
    for year in years:
        items = search_sentinel2(bbox, f"{year}-01-01/{year}-12-31", max_cloud=max_cloud)
        if not items:
            continue
        table = items_table(items)
        raw = load_raw_cube(items, bbox, crs=crs, resolution=resolution)
        cube = build_ndvi_cube(raw, table)                    # Session 1 decisions: offset, mask, duplicates
        x, y = cube["x"], cube["y"]
        months = pd.DatetimeIndex(cube["time"]).to_period("M")
        for month in sorted(set(months)):
            block = cube["ndvi"][months == month]
            layers.append(np.nanmedian(block, axis=0).astype("float32"))   # median of the clear days of the month
            n_obs.append(np.isfinite(block).sum(axis=0))
            times.append(month.to_timestamp())
        print(f"{year}: {len(items)} items, {len(cube['time'])} days")
    ndvi = np.stack(layers)
    prov = pd.DataFrame({"date": pd.DatetimeIndex(times).strftime("%Y-%m-%d"),
                         "n_obs_median": [float(np.median(n)) for n in n_obs],
                         "clear_fraction": np.isfinite(ndvi).mean(axis=(1, 2))})
    return {"ndvi": ndvi, "time": pd.DatetimeIndex(times), "x": x, "y": y, "provenance": prov}


def annual_mean_cube(ndvi, dates, min_share=0.6, steps_per_year=None):
    """One mean value per pixel and year, only if the year has enough data.

    Parameters
    ----------
    ndvi : numpy array (time, y, x)
    dates : pandas.DatetimeIndex, one per layer
    min_share : float, optional (default 0.6)
        A pixel-year needs at least this share of its steps with a value
        (e.g. 60 % of the 23 MODIS composites, or of the 12 months).
    steps_per_year : int, optional
        Expected number of steps per year (23 for MODIS, 12 for months).
        Default: the largest number of layers found in one year.

    Returns
    -------
    annual : numpy array (years, y, x)
    years : numpy array of int
    """
    dates = pd.DatetimeIndex(dates)
    years = np.unique(dates.year)
    if steps_per_year is None:
        steps_per_year = max((dates.year == yr).sum() for yr in years)
    annual = np.full((len(years),) + ndvi.shape[1:], np.nan)
    for k, yr in enumerate(years):
        sel = dates.year == yr
        share = np.isfinite(ndvi[sel]).sum(axis=0) / steps_per_year
        annual[k] = np.where(share >= min_share, np.nanmean(ndvi[sel], axis=0), np.nan)
    return annual, years

# =============================================================================
# PART 6 · CLIMATOLOGY, ANOMALIES, DECOMPOSITION AND TRENDS  (Session 2)
# =============================================================================
# All functions below work on ONE series (a numpy array or a pandas Series)
# unless the name says "cube" or "map". Missing values are NaN.

# ---- 6.1 The series as a table: years x periods ----------------------------------

def year_period_table(series, dates, periods=None):
    """Reshape a series into a table with one ROW per year and one COLUMN per period.

    It is the same idea as "the cube as a table" of Session 1, applied to one
    pixel: reading a row = one year; reading a column = the same moment of the
    year in every year. Climatology, anomalies and the seasonal Mann-Kendall test
    are all computed on this table.

    Parameters
    ----------
    series : array-like
        The values, one per date.
    dates : pandas.DatetimeIndex
        The date of each value.
    periods : array-like of int, optional
        The period of each date inside its year. Default: the MODIS composite
        number (``modis_period``: 1-23 for 16-day steps, 1-46 for Terra + Aqua).
        For half-months use 1 ... 24.

    Returns
    -------
    pandas.DataFrame, index = year, columns = period, values = the series.
    Missing combinations are NaN.
    """
    dates = pd.DatetimeIndex(dates)
    if periods is None:
        periods = modis_period(dates)
    df = pd.DataFrame({"year": dates.year, "period": np.asarray(periods),
                       "value": np.asarray(series, dtype="float64")})
    return df.pivot_table(index="year", columns="period", values="value", aggfunc="mean", dropna=False)


# ---- 6.2 Moving averages (Hyndman & Athanasopoulos, FPP3, section 3.3) ------------

def moving_average(x, m):
    """Centred moving average of ODD order m (an "m-MA").

    Each value is replaced by the mean of itself and its (m-1)/2 neighbours on
    each side:  T_t = (1/m) * sum_{j=-k..k} y_{t+j},  with m = 2k + 1.
    The first and last k values cannot be computed: they are NaN.

    Parameters
    ----------
    x : array-like
        The series (no missing values inside the window, or the result is NaN).
    m : int
        Order of the average (ODD): how many values are averaged.
        Larger m = smoother curve.

    Returns
    -------
    numpy array, same length as x.
    """
    if m % 2 == 0:
        raise ValueError("m must be odd for a centred m-MA; for even m use ma_2xm.")
    return weighted_moving_average(x, np.full(m, 1.0 / m))


def ma_2xm(x, m):
    """Centred "2 x m-MA": a moving average of EVEN order m, made symmetric.

    An average of an even number of values has no central value. The fix is to
    average two consecutive m-MA's (a "moving average of a moving average").
    The result is a weighted average of m + 1 values: weight 1/(2m) for the two
    end values and 1/m for all the others. Example, m = 4: [1/8, 1/4, 1/4, 1/4, 1/8].
    With m = number of periods per year (e.g. 24 half-months), the 2 x m-MA
    removes the seasonal cycle completely: this is the trend step of the
    classical decomposition.

    Parameters
    ----------
    x : array-like
        The series.
    m : int
        Order of the average (EVEN).

    Returns
    -------
    numpy array, same length as x (NaN at the m/2 values of each end).
    """
    if m % 2 == 1:
        raise ValueError("m must be even for a 2 x m-MA; for odd m use moving_average.")
    w = np.full(m + 1, 1.0 / m)
    w[0] = w[-1] = 1.0 / (2 * m)
    return weighted_moving_average(x, w)


def weighted_moving_average(x, weights):
    """Centred weighted moving average: T_t = sum_j a_j * y_{t+j}.

    Parameters
    ----------
    x : array-like
        The series.
    weights : array-like
        The weights a_{-k} ... a_{k}: an ODD number of them, SYMMETRIC
        (a_j = a_{-j}) and summing to 1. Examples: an m-MA has all weights 1/m;
        a 2 x 4-MA has [1/8, 1/4, 1/4, 1/4, 1/8]; a Savitzky-Golay filter with
        window 5 and order 2 has [-3, 12, 17, 12, -3] / 35 (Session 1).

    Returns
    -------
    numpy array, same length as x. NaN where the window does not fit or holds a NaN.
    """
    x = np.asarray(x, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    if len(w) % 2 == 0:
        raise ValueError("The number of weights must be odd (a centred average).")
    if not np.allclose(w, w[::-1]):
        raise ValueError("The weights must be symmetric (a_j = a_-j).")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError(f"The weights must sum to 1 (they sum to {w.sum():.4f}).")
    k = len(w) // 2
    out = np.full(len(x), np.nan)
    for t in range(k, len(x) - k):
        out[t] = np.dot(w, x[t - k:t + k + 1])          # a NaN anywhere in the window gives NaN
    return out


# ---- 6.3 Classical decomposition (FPP3, section 3.4) --------------------------------

def classical_decomposition(x, period, model="additive"):
    """Classical decomposition of a regular series into trend, seasonal and remainder.

    Steps (additive model, y = T + S + R):
        1. Trend-cycle T with a moving average of one year: an m-MA if the
           period is odd (MODIS: 23), a 2 x m-MA if it is even (half-months: 24).
        2. Detrended series: y - T.
        3. Seasonal component S: the mean of the detrended values of each period
           (all the "period 5" values together, and so on), adjusted so that the
           23 or 24 values add up to zero; then repeated every year.
        4. Remainder R = y - T - S.
    The multiplicative model (y = T x S x R) uses divisions instead.

    Known weaknesses (FPP3): no trend at the first and last half year; rapid rises
    and falls are over-smoothed; the seasonal pattern is forced to be IDENTICAL every
    year; one extreme year distorts everything (not robust).

    Parameters
    ----------
    x : array-like
        A REGULAR series with no missing values (fill the gaps first).
    period : int
        Number of observations per year (23 for MODIS, 24 for half-months).
    model : str, optional ("additive" or "multiplicative")

    Returns
    -------
    pandas.DataFrame with the columns "observed", "trend", "seasonal", "remainder".
    """
    x = np.asarray(x, dtype="float64")
    if np.isnan(x).any():
        raise ValueError("The series has missing values: fill them before decomposing.")
    trend = moving_average(x, period) if period % 2 else ma_2xm(x, period)
    detrended = x - trend if model == "additive" else x / trend
    position = np.arange(len(x)) % period                  # 0 ... period-1: the period of each value
    means = np.array([np.nanmean(detrended[position == p]) for p in range(period)])
    if model == "additive":
        means = means - means.mean()                       # seasonal values add up to zero
        seasonal = means[position]
        remainder = x - trend - seasonal
    else:
        means = means / means.mean()                       # seasonal indices average 1
        seasonal = means[position]
        remainder = x / (trend * seasonal)
    return pd.DataFrame({"observed": x, "trend": trend, "seasonal": seasonal, "remainder": remainder})


# ---- 6.4 Climatology and anomalies ---------------------------------------------------

def climatology(table, ref_years=None, stat="median", q=(10, 90)):
    """The "typical year" of a series: its central value and range for each period.

    Parameters
    ----------
    table : pandas.DataFrame
        The years x periods table (``year_period_table``).
    ref_years : tuple (first, last), optional
        Reference period, e.g. (2001, 2020). Default: all the years in the table.
        Declare it: anomalies are always "relative to" this period.
    stat : str, optional ("median" or "mean")
        Central value. The median is robust to one extreme year.
    q : tuple of 2 numbers, optional (default (10, 90))
        Percentiles of the "normal range" (the grey band of the figures).

    Returns
    -------
    pandas.DataFrame, one row per period, columns:
        "centre", "low", "high" (the percentiles), "spread" (robust standard
        deviation = 1.4826 x MAD, or the standard deviation when stat = "mean"),
        "n_years" (years with data in that period).
    """
    ref = table if ref_years is None else table.loc[ref_years[0]:ref_years[1]]
    if stat == "median":
        centre = ref.median()
        spread = 1.4826 * (ref - centre).abs().median()   # robust standard deviation (as Hampel, Session 1)
    else:
        centre = ref.mean()
        spread = ref.std()
    return pd.DataFrame({"centre": centre, "low": ref.quantile(q[0] / 100), "high": ref.quantile(q[1] / 100),
                         "spread": spread, "n_years": ref.notna().sum()})


def anomalies(table, clim, standardized=False, min_spread=0.01):
    """Anomaly = observed minus expected for that moment of the year.

    Parameters
    ----------
    table : pandas.DataFrame
        The years x periods table.
    clim : pandas.DataFrame
        The output of ``climatology``.
    standardized : bool, optional (default False)
        False -> absolute anomaly, in NDVI units: y - centre.
        True  -> standardised anomaly ("z"): (y - centre) / spread. It says how
        unusual a value is for THAT pixel and THAT period, so pixels with very
        different variability can be compared.
    min_spread : float, optional (default 0.01)
        Smallest spread allowed, to avoid dividing by almost zero in very stable
        periods.

    Returns
    -------
    pandas.DataFrame with the same shape as `table`.
    """
    a = table - clim["centre"]
    if standardized:
        a = a / np.maximum(clim["spread"], min_spread)
    return a


def table_to_series(table):
    """Undo ``year_period_table``: the values back in time order (year by year).

    Returns a numpy array of length years x periods (NaN where there was no data).
    """
    return table.values.ravel()


def climatology_cube(ndvi, dates, ref_years=None, periods=None):
    """Per-pixel climatology of a whole cube: median and robust spread per period.

    Parameters
    ----------
    ndvi : numpy array (time, y, x)
    dates : pandas.DatetimeIndex, one per layer
    ref_years : tuple (first, last), optional. Default: all years.
    periods : array of int, optional. Default: MODIS composite number.

    Returns
    -------
    centre, spread : numpy arrays (n_periods, y, x), with periods 1..P in order.
    """
    dates = pd.DatetimeIndex(dates)
    per = modis_period(dates) if periods is None else np.asarray(periods)
    in_ref = np.ones(len(dates), bool) if ref_years is None else \
        (dates.year >= ref_years[0]) & (dates.year <= ref_years[1])
    P = int(per.max())
    centre = np.full((P,) + ndvi.shape[1:], np.nan, dtype="float32")
    spread = np.full_like(centre, np.nan)
    for p in range(1, P + 1):
        block = ndvi[(per == p) & in_ref]
        if len(block):
            c = np.nanmedian(block, axis=0)
            centre[p - 1] = c
            spread[p - 1] = 1.4826 * np.nanmedian(np.abs(block - c), axis=0)
    return centre, spread


def anomaly_cube(ndvi, dates, centre, spread=None, periods=None, min_spread=0.01):
    """Anomaly of every layer of a cube against its per-pixel climatology.

    Parameters
    ----------
    ndvi : numpy array (time, y, x)
    dates : pandas.DatetimeIndex
    centre, spread : outputs of ``climatology_cube``. If `spread` is given, the
        result is the STANDARDISED anomaly; otherwise the absolute anomaly.
    periods, min_spread : as in ``climatology_cube`` and ``anomalies``.

    Returns
    -------
    numpy array (time, y, x).
    """
    per = modis_period(dates) if periods is None else np.asarray(periods)
    a = ndvi - centre[per - 1]
    if spread is not None:
        a = a / np.maximum(spread[per - 1], min_spread)
    return a


# ---- 6.5 One value per year ------------------------------------------------------------

def annual_aggregate(table, how="mean", min_share=0.7, step_days=None):
    """Summarise each year in one number (one ROW of the years x periods table).

    Parameters
    ----------
    table : pandas.DataFrame
        The years x periods table.
    how : str, optional
        "mean"     -> mean NDVI of the year;
        "max"      -> the peak of the year;
        "integral" -> area under the NDVI curve ABOVE the base level of the
                      year (its 10th percentile, the bare-soil or dry-season
                      value), in NDVI-days. It measures the seasonal growth:
                      how much vegetation grew, and for how long (the "small
                      integral" of TIMESAT, Jönsson & Eklundh, 2004). It is
                      the bridge between the shape of the season (intra-annual)
                      and the years (inter-annual).
    min_share : float, optional (default 0.7)
        A year with fewer than this share of periods with data gets NaN: a year
        with half its values missing does not describe the whole year.
    step_days : number, optional
        Days represented by each period. Default: 365.25 / number of periods
        (about 16 for 23 composites, 8 for 46, 15 for 24 half-months).

    Returns
    -------
    pandas.Series indexed by year.
    """
    share = table.notna().mean(axis=1)
    if how == "mean":
        out = table.mean(axis=1)
    elif how == "max":
        out = table.max(axis=1)
    elif how == "integral":
        # missing periods are interpolated from their neighbours, so that a missing
        # value does not count as "zero vegetation"
        filled = table.apply(lambda row: row.interpolate(limit_direction="both"), axis=1)
        base = filled.quantile(0.10, axis=1)                    # base level of each year
        days = step_days if step_days is not None else 365.25 / table.shape[1]
        out = filled.sub(base, axis=0).clip(lower=0).sum(axis=1) * days
    else:
        raise ValueError("how must be 'mean', 'max' or 'integral'")
    return out.where(share >= min_share)


# ---- 6.6 Trend tests: Mann-Kendall and Sen -----------------------------------------------

def mann_kendall(x):
    """Mann-Kendall test for a MONOTONIC trend (Mann, 1945; Kendall, 1975).

    It compares every pair of values (i before j) and counts how many times the
    later one is larger minus how many times it is smaller:
        S = sum_{i<j} sign(x_j - x_i)
    With no trend and independent values, S is centred on 0 with variance
        Var(S) = [n(n-1)(2n+5) - sum_ties t(t-1)(2t+5)] / 18
    and Z = (S - sign(S)) / sqrt(Var(S)) follows a standard normal distribution.

    It assumes NOTHING about normality or linearity; it DOES assume that the
    values are independent (no autocorrelation) and have no seasonal cycle.

    Parameters
    ----------
    x : array-like
        The series in time order. NaN values are dropped.

    Returns
    -------
    dict with "n", "S", "var_S", "Z", "p" (two-sided), "tau" (Kendall's tau, -1..1)
    and "trend" ("increasing", "decreasing" or "no trend" at alpha = 0.05).
    """
    from scipy.stats import norm
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return {"n": n, "S": np.nan, "var_S": np.nan, "Z": np.nan, "p": np.nan, "tau": np.nan, "trend": "too short"}
    diff = np.sign(x[None, :] - x[:, None])               # diff[i, j] = sign(x_j - x_i)
    S = np.triu(diff, k=1).sum()                           # only pairs with i < j
    _, counts = np.unique(x, return_counts=True)           # groups of tied values
    ties = counts[counts > 1]
    var_S = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    Z = 0.0 if S == 0 else (S - np.sign(S)) / np.sqrt(var_S)
    p = 2 * (1 - norm.cdf(abs(Z)))
    tau = S / (n * (n - 1) / 2)
    trend = "no trend" if p >= 0.05 else ("increasing" if S > 0 else "decreasing")
    return {"n": n, "S": float(S), "var_S": float(var_S), "Z": float(Z), "p": float(p), "tau": float(tau), "trend": trend}


def sens_slope(x, t=None, alpha=0.05):
    """Theil-Sen slope: the MEDIAN of the slopes between all pairs of values.

        beta = median{ (x_j - x_i) / (t_j - t_i) : i < j }

    Because it is a median, a few extreme values (an El Nino year, a residual
    cloud) barely move it.

    Parameters
    ----------
    x : array-like
        The values.
    t : array-like, optional
        The time of each value (e.g. the year). Default: 0, 1, 2, ...
        The slope is in units of x per unit of t (e.g. NDVI per year).
    alpha : float, optional (default 0.05)
        For the confidence interval of the slope (Gilbert, 1987, method).

    Returns
    -------
    dict with "slope", "intercept" (so that x ~ intercept + slope * t),
    "low" and "high" (confidence interval of the slope).
    """
    from scipy.stats import norm
    x = np.asarray(x, dtype="float64")
    t = np.arange(len(x), dtype="float64") if t is None else np.asarray(t, dtype="float64")
    ok = ~np.isnan(x)
    x, t = x[ok], t[ok]
    n = len(x)
    if n < 3:
        return {"slope": np.nan, "intercept": np.nan, "low": np.nan, "high": np.nan}
    i, j = np.triu_indices(n, k=1)
    slopes = np.sort((x[j] - x[i]) / (t[j] - t[i]))
    slope = np.median(slopes)
    intercept = np.median(x - slope * t)
    var_S = n * (n - 1) * (2 * n + 5) / 18.0
    c = norm.ppf(1 - alpha / 2) * np.sqrt(var_S)
    N = len(slopes)
    lo_i, hi_i = int(np.floor((N - c) / 2)), int(np.ceil((N + c) / 2))
    low = slopes[max(lo_i, 0)]
    high = slopes[min(hi_i, N - 1)]
    return {"slope": float(slope), "intercept": float(intercept), "low": float(low), "high": float(high)}


def mann_kendall_hamed_rao(x, alpha=0.05):
    """Mann-Kendall with the variance correction of Hamed & Rao (1998).

    Positive autocorrelation (each value resembles the previous one) means the
    series carries less independent information than its length suggests: the
    real variance of S is LARGER than the formula of the ordinary test, which
    therefore finds trends that do not exist. Hamed & Rao estimate how much
    larger, and inflate Var(S) accordingly:
        1. remove the Sen slope from the series;
        2. compute the autocorrelation of the RANKS of what is left, at every lag k;
        3. keep only the significant lags (|rho_k| > z_(1-alpha/2) / sqrt(n));
        4. correction factor
           n/n* = 1 + 2 / (n(n-1)(n-2)) * sum_k (n-k)(n-k-1)(n-k-2) rho_k
        5. Var*(S) = Var(S) * n/n*, and the test continues as usual.
    This is the method `hamed_rao_modification_test` of pyMannKendall.

    Parameters
    ----------
    x : array-like (time order, NaN dropped)
    alpha : float, optional (default 0.05): level used to select significant lags.

    Returns
    -------
    The dict of ``mann_kendall`` (with the corrected Var(S), Z and p), plus
    "correction" (the factor n/n*; > 1 when the series is positively autocorrelated).
    """
    from scipy.stats import norm, rankdata
    x = np.asarray(x, dtype="float64")
    x = x[~np.isnan(x)]
    n = len(x)
    out = mann_kendall(x)
    if n < 4:
        out["correction"] = 1.0
        return out
    t = np.arange(n)
    detr = x - sens_slope(x, t)["slope"] * t
    r = rankdata(detr) - (n + 1) / 2.0
    denom = np.sum(r * r)
    acf = np.array([np.sum(r[k:] * r[:-k]) / denom for k in range(1, n)])
    limit = norm.ppf(1 - alpha / 2) / np.sqrt(n)
    k = np.arange(1, n)
    keep = np.abs(acf) > limit
    factor = 1 + 2.0 / (n * (n - 1) * (n - 2)) * np.sum(((n - k) * (n - k - 1) * (n - k - 2) * acf)[keep])
    factor = max(factor, 1.0)
    var_S = out["var_S"] * factor
    S = out["S"]
    Z = 0.0 if S == 0 else (S - np.sign(S)) / np.sqrt(var_S)
    p = 2 * (1 - norm.cdf(abs(Z)))
    out.update({"var_S": float(var_S), "Z": float(Z), "p": float(p), "correction": float(factor),
                "trend": "no trend" if p >= 0.05 else ("increasing" if S > 0 else "decreasing")})
    return out


def seasonal_mann_kendall(table):
    """Seasonal Mann-Kendall test (Hirsch, Slack & Smith, 1982).

    The test is run SEPARATELY for each period of the year (all the "period 5"
    values across the years, and so on), and the S statistics and variances are
    added. A peak is never compared with a trough, so the seasonal cycle does
    not need to be removed first.

    Parameters
    ----------
    table : pandas.DataFrame
        The years x periods table.

    Returns
    -------
    dict with "S", "var_S", "Z", "p", "slope" (seasonal Sen slope: the median of
    the within-period slopes, in units per year) and "trend".
    """
    from scipy.stats import norm
    S_total, var_total, slopes = 0.0, 0.0, []
    years = table.index.values.astype("float64")
    for p in table.columns:
        col = table[p].values.astype("float64")
        ok = ~np.isnan(col)
        if ok.sum() < 3:
            continue
        r = mann_kendall(col[ok])
        S_total += r["S"]; var_total += r["var_S"]
        xv, tv = col[ok], years[ok]
        i, j = np.triu_indices(len(xv), k=1)
        slopes.extend(((xv[j] - xv[i]) / (tv[j] - tv[i])).tolist())
    if var_total == 0:
        return {"S": np.nan, "var_S": np.nan, "Z": np.nan, "p": np.nan, "slope": np.nan, "trend": "too short"}
    Z = 0.0 if S_total == 0 else (S_total - np.sign(S_total)) / np.sqrt(var_total)
    p = 2 * (1 - norm.cdf(abs(Z)))
    trend = "no trend" if p >= 0.05 else ("increasing" if S_total > 0 else "decreasing")
    return {"S": S_total, "var_S": var_total, "Z": float(Z), "p": float(p),
            "slope": float(np.median(slopes)), "trend": trend}


def rolling_sen(annual, window=10):
    """Sen slope in moving windows of `window` years: does the slope change?

    Parameters
    ----------
    annual : pandas.Series indexed by year (e.g. ``annual_aggregate``).
    window : int, optional (default 10)
        Years in each window. Shorter = more detail, but noisier.

    Returns
    -------
    pandas.Series indexed by the CENTRAL year of each window, values = slope
    per year. Neighbouring windows share most of their years, so this curve
    is smooth by construction: read it as a description, not as a test.
    """
    years = annual.index.values
    out = {}
    for k in range(len(years) - window + 1):
        part = annual.iloc[k:k + window]
        if part.notna().sum() >= max(3, int(0.7 * window)):
            centre = (years[k] + years[k + window - 1]) / 2
            out[centre] = sens_slope(part.values, part.index.values)["slope"]
    return pd.Series(out, name=f"Sen slope, {window}-year window")


# ---- 6.7 Maps of trends and the multiple-testing problem --------------------------------

def trend_map(annual_cube, years, method="hamed_rao", min_years=10):
    """Trend test and Sen slope for every pixel of a cube of annual values.

    Parameters
    ----------
    annual_cube : numpy array (years, y, x)
        One value per year and pixel (e.g. the annual mean or integral).
    years : array-like of int
        The year of each layer.
    method : str, optional ("hamed_rao" or "original")
        "hamed_rao" corrects the variance for autocorrelation (recommended);
        "original" is the plain Mann-Kendall test.
    min_years : int, optional (default 10)
        Pixels with fewer valid years get NaN.

    Returns
    -------
    slope, p : numpy arrays (y, x): Sen slope (units per year) and p-value.
    """
    test = mann_kendall_hamed_rao if method == "hamed_rao" else mann_kendall
    n_y, ny, nx = annual_cube.shape
    slope = np.full((ny, nx), np.nan)
    pval = np.full((ny, nx), np.nan)
    years = np.asarray(years, dtype="float64")
    for r in range(ny):
        for c in range(nx):
            s = annual_cube[:, r, c]
            if np.isfinite(s).sum() >= min_years:
                slope[r, c] = sens_slope(s, years)["slope"]
                pval[r, c] = test(s)["p"]
    return slope, pval


def fdr_mask(p, alpha=0.10):
    """Which p-values remain significant after controlling the False Discovery Rate.

    With thousands of pixels, a p < 0.05 threshold declares about 5 % of them
    "significant" even if nothing changes anywhere. The Benjamini-Hochberg
    procedure sorts the p-values and accepts only those with p_(k) <= (k/m) alpha.
    Wilks (2016) recommends it for maps, with alpha_FDR = 2 x the desired global
    level (alpha = 0.10 for a global 0.05).

    Parameters
    ----------
    p : numpy array (any shape), NaN allowed
    alpha : float, optional (default 0.10)

    Returns
    -------
    numpy array of bool, same shape: True = significant after FDR.
    """
    from scipy.stats import false_discovery_control
    p = np.asarray(p, dtype="float64")
    out = np.zeros(p.shape, bool)
    ok = np.isfinite(p)
    if ok.any():
        adjusted = false_discovery_control(p[ok], method="bh")
        out[ok] = adjusted <= alpha
    return out


# ---- 6.8 Relations in time and synthetic series ---------------------------------------------

def lag_correlation(a, b, max_lag=6):
    """Correlation between series a and series b shifted by 0 ... max_lag steps.

    Lag k compares a_t with b_{t-k}: does `a` (e.g. NDVI anomaly) respond to `b`
    (e.g. an ENSO index) k steps later?

    Parameters
    ----------
    a, b : pandas.Series with the SAME index (same dates).
    max_lag : int, optional (default 6)

    Returns
    -------
    pandas.Series indexed by lag, values = Pearson correlation.
    """
    return pd.Series({k: a.corr(b.shift(k)) for k in range(max_lag + 1)}, name="correlation")


def synthetic_ndvi(n_years=25, period=23, trend_per_year=0.0, amplitude=0.25, base=0.55,
                   interannual=0.03, ar1=0.0, noise=0.03, step_year=None, step_size=0.0, seed=0):
    """A synthetic NDVI series whose components we KNOW (to test the methods).

    y = base + seasonal cycle + trend + year effect (+ step) + AR(1) noise

    Parameters
    ----------
    n_years, period : int
        Length of the series: n_years x period values.
    trend_per_year : float
        The true trend, NDVI per year (0 = no trend).
    amplitude : float
        Half the height of the seasonal cycle.
    base : float
        Mean level of the series.
    interannual : float
        Standard deviation of a random shift of each whole year (wet/dry years).
    ar1 : float (0 ... <1)
        Lag-1 autocorrelation of the noise. 0 = independent noise. Smoothing and
        gap filling make real series autocorrelated.
    noise : float
        Standard deviation of the noise.
    step_year, step_size : optional
        Add an abrupt change (a step) of `step_size` from year `step_year` on.
    seed : int
        Seed of the random numbers (same seed = same series).

    Returns
    -------
    pandas.DataFrame with "date", "year", "period", "value" and the true
    components "seasonal", "trend", "year_effect", "noise".
    """
    rng = np.random.default_rng(seed)
    n = n_years * period
    k = np.arange(n)
    year = 2001 + k // period
    per = k % period + 1
    seasonal = amplitude * np.sin(2 * np.pi * (per - 1) / period - np.pi / 2)
    trend = trend_per_year * (k / period)
    year_effect = rng.normal(0, interannual, n_years)[k // period]
    if step_year is not None:
        year_effect = year_effect + np.where(year >= step_year, step_size, 0.0)
    e = np.zeros(n)
    eps = rng.normal(0, noise * np.sqrt(1 - ar1 ** 2), n)
    e[0] = rng.normal(0, noise)
    for i in range(1, n):
        e[i] = ar1 * e[i - 1] + eps[i]
    value = base + seasonal + trend + year_effect + e
    dates = pd.to_datetime([f"{y}-01-01" for y in year]) + pd.to_timedelta((per - 1) * 16, unit="D")
    return pd.DataFrame({"date": dates, "year": year, "period": per, "value": value,
                         "seasonal": seasonal, "trend": trend, "year_effect": year_effect, "noise": e})


# =============================================================================
# PART 7 · HELPERS FOR THE SESSION 2 NOTEBOOKS  (choices and figures used many times)
# =============================================================================
# These functions do NOT add new ideas: they hide repetitive code (building a
# file once, drawing red/green bars, a map without axes) so that each notebook
# cell shows only the idea of that step.

def get_modis_file(path, bbox, period, crs="EPSG:32618", platform="both", title="MODIS NDVI"):
    """Return the MODIS cube saved in `path`; build and save it first if the file does not exist.

    Parameters
    ----------
    path : str
        Name of the HDF5 file (e.g. "saldana_modis_ndvi_2001_2015_terra_aqua.h5").
    bbox : list [west, south, east, north] in degrees
    period : str "YYYY-MM-DD/YYYY-MM-DD"
    crs : str, optional, the projection of the cube (metres)
    platform : "terra", "aqua" or "both" (both = one composite every 8 days)
    title : str, written into the file

    Returns
    -------
    dict with "ndvi" (time, y, x), "time", "x", "y", "provenance", "attrs"
    """
    import os
    if not os.path.exists(path):
        print("The file does not exist: building it from Planetary Computer (several minutes)...")
        cube = load_modis_cube(bbox, period, crs=crs, resolution=250, platform=platform, good_classes=(0, 1))
        attrs = {"crs": crs, "resolution_m": 250.0, "bbox_wgs84": str(bbox), "period": period,
                 "source": "MOD13Q1 (Terra) + MYD13Q1 (Aqua) v061, Microsoft Planetary Computer",
                 "mask_rule": "pixel_reliability in (0, 1) kept; 2 (snow) and 3 (cloudy) -> NaN",
                 "scale_rule": "NDVI = DN x 0.0001; DN = -3000 -> NaN",
                 "time_axis": "regular axis; a missing composite is a layer of NaN"}
        n_years = len(np.unique(pd.DatetimeIndex(cube["time"]).year))
        save_cube_hdf5(path, cube, chunks=(min(len(cube["time"]), 46 * n_years), 20, 20), attrs=attrs, title=title)
    return load_cube_hdf5(path)


def get_s2_monthly_file(path, bbox, years, crs="EPSG:32618", resolution=20):
    """Return the monthly Sentinel-2 cube saved in `path`; build it first if needed (slow: minutes per year)."""
    import os
    if not os.path.exists(path):
        print("The file does not exist: building it with the chain of Session 1 (several minutes per year)...")
        s2 = build_s2_monthly_cube(bbox, years, crs=crs, resolution=resolution)
        save_cube_hdf5(path, s2, chunks=(96, 40, 40), title="Sentinel-2 L2A monthly median NDVI",
                       attrs={"crs": crs, "resolution_m": float(resolution), "bbox_wgs84": str(bbox),
                              "chain": "Session 1: offset, SCL 4-5-6 kept, mask before fusing duplicates",
                              "monthly_rule": "median of the clear values of each month"})
    return load_cube_hdf5(path)


def _local_std(image, size=3):
    """Standard deviation of each pixel's size x size neighbourhood (NaN-aware): high = mixed or edge pixel."""
    k = size // 2
    padded = np.pad(np.asarray(image, dtype="float64"), k, mode="constant", constant_values=np.nan)
    stack = [padded[i:i + image.shape[0], j:j + image.shape[1]] for i in range(size) for j in range(size)]
    return np.nanstd(np.stack(stack), axis=0)


def candidate_pixels(ndvi, dates, n=5, min_share=0.6, min_distance=3):
    """Several candidate pixels per class, chosen from the TYPICAL YEAR of each pixel.

    The typical year (median of each composite over all years) removes noise and
    single bad years, so the classes are judged by their seasonal behaviour:

    * "seasonal crop"     : large amplitude of the typical year (irrigated rice: green, then bare);
    * "stable vegetation" : high level, small amplitude and small year-to-year spread (riparian forest);
    * "intermediate"      : level and amplitude close to the middle of the window (pastures, rain-fed land).

    In every class, pixels whose 3 x 3 neighbourhood is heterogeneous (edges, mixed
    pixels) are penalised, and candidates are kept at least `min_distance` pixels
    apart. Candidates are ordered from best to worst score: check them on a map
    (`map_pixels`) before trusting them.

    Parameters
    ----------
    ndvi : numpy array (time, rows, columns)
    dates : pandas.DatetimeIndex, one per layer
    n : int, number of candidates per class
    min_share : float, minimum share of layers with a value
    min_distance : int, minimum distance in pixels between candidates of a class

    Returns
    -------
    candidates : dict  class -> list of (row, col), best first
    summary : pandas.DataFrame, one row per candidate: class, rank, row, col, level, amplitude, spread, neighbourhood
    """
    dates = pd.DatetimeIndex(dates)
    centre, spread = climatology_cube(ndvi, dates)
    with np.errstate(all="ignore"):
        level = np.nanmedian(centre, axis=0)
        amp = np.nanpercentile(centre, 95, axis=0) - np.nanpercentile(centre, 5, axis=0)
        yearly_spread = np.nanmedian(spread, axis=0)
        het = _local_std(level)
    well = np.isfinite(ndvi).mean(axis=0) >= min_share
    middle_level, middle_amp = np.nanmedian(level[well]), np.nanmedian(amp[well])
    scores = {
        "seasonal crop": np.where(well & (level > 0.3), amp - 2 * het, np.nan),
        "stable vegetation": np.where(well & (amp < np.nanpercentile(amp[well], 30)), level - amp - yearly_spread - 2 * het, np.nan),
        "intermediate": np.where(well, -np.abs(level - middle_level) - np.abs(amp - middle_amp) - 2 * het, np.nan),
    }
    candidates, rows = {}, []
    for name, score in scores.items():
        order = np.argsort(np.where(np.isfinite(score), -score, np.inf), axis=None)
        chosen = []
        for flat in order:
            r, c = np.unravel_index(flat, score.shape)
            if not np.isfinite(score[r, c]):
                break
            if all(max(abs(r - r2), abs(c - c2)) >= min_distance for r2, c2 in chosen):
                chosen.append((int(r), int(c)))
            if len(chosen) == n:
                break
        candidates[name] = chosen
        for k, (r, c) in enumerate(chosen, start=1):
            rows.append({"class": name, "candidate": k, "row": r, "col": c, "level": round(float(level[r, c]), 2),
                         "amplitude": round(float(amp[r, c]), 2), "year_to_year_spread": round(float(yearly_spread[r, c]), 3),
                         "neighbourhood_std": round(float(het[r, c]), 3)})
    return candidates, pd.DataFrame(rows)


def example_pixels(ndvi, dates=None, min_share=0.6):
    """The best candidate of each class (see `candidate_pixels`): dict class -> (row, col).

    If `dates` is not given, a simpler rule on the raw values is used (90th - 10th percentile).
    """
    if dates is not None:
        candidates, _ = candidate_pixels(ndvi, dates, n=1, min_share=min_share)
        return {name: cands[0] for name, cands in candidates.items()}
    n_valid = np.isfinite(ndvi).sum(axis=0)
    p10, p50, p90 = (np.nanpercentile(ndvi, q, axis=0) for q in (10, 50, 90))
    amp = p90 - p10
    well = n_valid >= min_share * ndvi.shape[0]

    def best(score):
        r, c = np.unravel_index(np.nanargmax(score), score.shape)
        return int(r), int(c)

    middle = -np.abs(p50 - np.nanmedian(p50)) - np.abs(amp - np.nanmedian(amp))
    return {"seasonal crop": best(np.where(well & (p50 > 0.35), amp, np.nan)),
            "stable vegetation": best(np.where(well & (amp < np.nanpercentile(amp, 25)), p50, np.nan)),
            "intermediate": best(np.where(well, middle, np.nan))}


def complete_series(values, dates, period=46, max_gap=2):
    """A series WITHOUT holes, for moving averages and decompositions.

    1) short gaps (<= max_gap steps) are filled linearly in time (Session 1);
    2) longer gaps are filled with the median of that composite over all years
       ("a normal value for that time of year"). Values filled in step 2 are normal
       BY CONSTRUCTION: never use them to look for anomalies or trends.

    Returns
    -------
    filled : numpy array
    n_long : int, number of values filled in step 2
    """
    s = pd.DataFrame({"v": np.asarray(values, dtype="float64")}, index=pd.DatetimeIndex(dates))
    filled, _ = fill_gaps(s, max_gap=max_gap)
    out = filled["v"].values.copy()
    table = year_period_table(out, dates)
    typical = table.median()
    position = np.arange(len(out)) % period
    holes = np.isnan(out)
    out[holes] = typical.values[position[holes]]
    return out, int(holes.sum())


def bars(ax, x, values, width=0.8):
    """Red bars below zero, green bars above zero (anomalies, slopes...)."""
    values = np.asarray(values, dtype="float64")
    ax.bar(x, np.where(values > 0, values, 0), width=width, color="#2B580C")
    ax.bar(x, np.where(values < 0, values, 0), width=width, color="#C0392B")
    ax.axhline(0, color="k", lw=0.6)


def shade_enso(ax, episodes):
    """Shade ENSO episodes: list of (kind, "YYYY-MM", "YYYY-MM"); orange = El Niño, blue = La Niña."""
    for kind, start, end in episodes:
        colour = "#F2C29B" if kind == "El Niño" else "#BBD5E8"
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=colour, alpha=0.5, lw=0)


def show_map(ax, image, title="", cmap="RdYlGn", vmin=0, vmax=0.9):
    """A map without axes or grid. Returns the image (for the colour bar)."""
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return im


def show_trend_map(ax, slope, p=None, title="", lim=None, alpha=0.10, mask=None):
    """Sen slope per DECADE in brown-green, with black dots where the trend is significant.

    The dots come from `mask` if it is given (a boolean map); otherwise from the
    FDR correction of the p-values `p` (Benjamini-Hochberg, `alpha`)."""
    per_decade = 10 * slope
    if lim is None:
        lim = np.nanpercentile(np.abs(per_decade), 98)
    im = show_map(ax, per_decade, title, cmap="BrBG", vmin=-lim, vmax=lim)
    if mask is None:
        mask = fdr_mask(p, alpha=alpha)
    rows, cols = np.where(mask)
    ax.plot(cols, rows, ".", ms=2, color="k")
    return im


def compare_terra_aqua(ndvi, dates, platform):
    """Window-mean NDVI of each Aqua composite next to the Terra composite of 8 days before.

    Returns
    -------
    pandas.DataFrame with columns "terra_8_days_before", "aqua" (one row per pair)
    """
    dates = pd.DatetimeIndex(dates)
    mean = pd.Series(np.nanmean(ndvi, axis=(1, 2)), index=dates)
    aqua = mean[np.asarray(platform) == "aqua"].dropna()
    terra = mean.reindex(aqua.index - pd.Timedelta(days=8))
    return pd.DataFrame({"terra_8_days_before": terra.values, "aqua": aqua.values}, index=aqua.index).dropna()


def pixels_inside(big, small):
    """Rows and columns of the cube `big` whose centres fall inside the extent of the cube `small`
    (both with "x" and "y" in the same CRS, metres). Returns two boolean arrays."""
    cols = (big["x"] >= np.min(small["x"])) & (big["x"] <= np.max(small["x"]))
    rows = (big["y"] >= np.min(small["y"])) & (big["y"] <= np.max(small["y"]))
    return rows, cols


def false_positive_experiment(ar1, interannual=0.0, trend=0.0, n_series=300, n_years=15, alpha=0.05):
    """Share of synthetic series called 'significant' by three ways of testing.

    Each series has `n_years` years, one composite every 16 days, noise with lag-1
    autocorrelation `ar1`, wet/dry years with standard deviation `interannual`, and a
    TRUE trend `trend` (NDVI per year; 0 = no trend). Three ways of testing:
    "original" Mann-Kendall on all the anomalies; "Hamed-Rao" on all the anomalies;
    "annual" Mann-Kendall on one mean value per year.

    Returns
    -------
    dict  method -> share of series with p < alpha
    """
    hits = {"original": 0, "Hamed-Rao": 0, "annual": 0}
    for seed in range(n_series):
        d = synthetic_ndvi(n_years=n_years, trend_per_year=trend, ar1=ar1, interannual=interannual, noise=0.03, seed=seed)
        table = year_period_table(d["value"].values, d["date"])
        anomaly = table_to_series(table - table.median())
        hits["original"] += mann_kendall(anomaly)["p"] < alpha
        hits["Hamed-Rao"] += mann_kendall_hamed_rao(anomaly)["p"] < alpha
        hits["annual"] += mann_kendall(annual_aggregate(table, "mean").values)["p"] < alpha
    return {k: v / n_series for k, v in hits.items()}


# -----------------------------------------------------------------------------
# Checking pixels on a high-resolution image  (Session 2, Part A, section A1)
# -----------------------------------------------------------------------------
# A MODIS pixel is 250 x 250 m: it can mix a paddy, a road and a canal. Before
# trusting "the rice pixel", we look at it on a high-resolution image. The map
# needs the library folium (conda install -c conda-forge folium).

def _utm_zone(crs):
    """Zone number and hemisphere of a WGS 84 / UTM code: "EPSG:32618" -> (18, "N")."""
    code = int(str(crs).upper().replace("EPSG:", ""))
    if 32601 <= code <= 32660:
        return code - 32600, "N"
    if 32701 <= code <= 32760:
        return code - 32700, "S"
    raise ValueError(f"{crs} is not a WGS 84 / UTM projection (EPSG:326xx or EPSG:327xx)")


_A, _F, _K0 = 6378137.0, 1 / 298.257223563, 0.9996          # WGS 84 ellipsoid, UTM scale factor
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def utm_to_lonlat(x, y, crs="EPSG:32618"):
    """UTM metres -> (longitude, latitude) in degrees (WGS 84). Accurate to well under 1 m.

    Uses the standard series of the transverse Mercator projection, so no extra
    library is needed. `x`, `y` can be numbers or numpy arrays.
    """
    zone, hemi = _utm_zone(crs)
    x = np.asarray(x, dtype="float64") - 500000.0
    y = np.asarray(y, dtype="float64") - (10000000.0 if hemi == "S" else 0.0)
    e1 = (1 - np.sqrt(1 - _E2)) / (1 + np.sqrt(1 - _E2))
    mu = (y / _K0) / (_A * (1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256))
    phi1 = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    n1 = _A / np.sqrt(1 - _E2 * np.sin(phi1) ** 2)
    t1 = np.tan(phi1) ** 2
    c1 = _EP2 * np.cos(phi1) ** 2
    r1 = _A * (1 - _E2) / (1 - _E2 * np.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * _K0)
    lat = phi1 - (n1 * np.tan(phi1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * _EP2) * d ** 4 / 24
                                             + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * _EP2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * _EP2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(phi1)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    return np.degrees(lon + lon0), np.degrees(lat)


def lonlat_to_utm(lon, lat, crs="EPSG:32618"):
    """(longitude, latitude) in degrees -> UTM metres (x, y). The inverse of `utm_to_lonlat`."""
    zone, hemi = _utm_zone(crs)
    phi = np.radians(np.asarray(lat, dtype="float64"))
    lam = np.radians(np.asarray(lon, dtype="float64")) - np.radians((zone - 1) * 6 - 180 + 3)
    n = _A / np.sqrt(1 - _E2 * np.sin(phi) ** 2)
    t = np.tan(phi) ** 2
    c = _EP2 * np.cos(phi) ** 2
    a = np.cos(phi) * lam
    m = _A * ((1 - _E2 / 4 - 3 * _E2 ** 2 / 64 - 5 * _E2 ** 3 / 256) * phi
              - (3 * _E2 / 8 + 3 * _E2 ** 2 / 32 + 45 * _E2 ** 3 / 1024) * np.sin(2 * phi)
              + (15 * _E2 ** 2 / 256 + 45 * _E2 ** 3 / 1024) * np.sin(4 * phi) - (35 * _E2 ** 3 / 3072) * np.sin(6 * phi))
    x = _K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t ** 2 + 72 * c - 58 * _EP2) * a ** 5 / 120) + 500000.0
    y = _K0 * (m + n * np.tan(phi) * (a ** 2 / 2 + (5 - t + 9 * c + 4 * c ** 2) * a ** 4 / 24
                                      + (61 - 58 * t + t ** 2 + 600 * c - 330 * _EP2) * a ** 6 / 720))
    if hemi == "S":
        y = y + 10000000.0
    return x, y


def _cube_crs(cube, crs=None):
    return crs or cube.get("attrs", {}).get("crs", "EPSG:32618")


def pixel_to_lonlat(cube, row, col, crs=None):
    """Centre of pixel (row, col) of the cube, as (longitude, latitude)."""
    lon, lat = utm_to_lonlat(cube["x"][col], cube["y"][row], _cube_crs(cube, crs))
    return float(lon), float(lat)


def lonlat_to_pixel(cube, lon, lat, crs=None):
    """The (row, col) of the cube pixel that contains the point (longitude, latitude).

    Raises an error if the point falls outside the cube.
    """
    x, y = lonlat_to_utm(lon, lat, _cube_crs(cube, crs))
    xs, ys = np.asarray(cube["x"]), np.asarray(cube["y"])
    half_x, half_y = abs(xs[1] - xs[0]) / 2, abs(ys[1] - ys[0]) / 2
    col, row = int(np.argmin(np.abs(xs - x))), int(np.argmin(np.abs(ys - y)))
    if abs(xs[col] - x) > half_x or abs(ys[row] - y) > half_y:
        raise ValueError(f"The point ({lat:.5f}, {lon:.5f}) is outside the cube window.")
    return row, col


def pixel_footprint(cube, row, col, crs=None):
    """The four corners of pixel (row, col) as [lat, lon] pairs (for drawing it on a map)."""
    xs, ys = np.asarray(cube["x"]), np.asarray(cube["y"])
    hx, hy = abs(xs[1] - xs[0]) / 2, abs(ys[1] - ys[0]) / 2
    cx, cy = xs[col], ys[row]
    corners_x = np.array([cx - hx, cx + hx, cx + hx, cx - hx])
    corners_y = np.array([cy + hy, cy + hy, cy - hy, cy - hy])
    lon, lat = utm_to_lonlat(corners_x, corners_y, _cube_crs(cube, crs))
    return [[float(a), float(b)] for a, b in zip(lat, lon)]


def map_pixels(cube, pixels, colours=None, background=None, zoom=14, crs=None):
    """An interactive map (folium) to check the chosen pixels on a high-resolution image.

    Layers: a satellite image (Esri World Imagery, recent: NOT from 2001-2015),
    OpenStreetMap, and optionally `background` (e.g. the median NDVI map) drawn
    semi-transparent over the window. Each pixel is drawn as its real 250 m square.
    Click anywhere on the map to see its latitude and longitude (to choose your
    own pixels with `lonlat_to_pixel`).

    Parameters
    ----------
    cube : dict with "x", "y" (and "attrs" with "crs")
    pixels : dict name -> (row, col); can be empty {} to see only the window
    colours : dict name -> colour, optional
    background : 2-D array (rows, columns) of the cube, optional
    zoom : int, initial zoom (14 shows a few km)

    Returns
    -------
    folium.Map  (write the variable on the last line of a cell to display it)
    """
    import folium
    from folium.raster_layers import ImageOverlay
    crs = _cube_crs(cube, crs)
    colours = colours or {}
    pixels = pixels or {}
    if pixels:                                               # centre the map on the first pixel...
        lon0, lat0 = pixel_to_lonlat(cube, *next(iter(pixels.values())), crs=crs)
    else:                                                    # ...or on the centre of the window
        lon0, lat0 = utm_to_lonlat(np.mean(cube["x"]), np.mean(cube["y"]), crs)
        lon0, lat0 = float(lon0), float(lat0)
        zoom = min(zoom, 13)
    m = folium.Map(location=[lat0, lon0], zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics", name="Satellite image (Esri, recent)").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    xs, ys = np.asarray(cube["x"]), np.asarray(cube["y"])
    hx, hy = abs(xs[1] - xs[0]) / 2, abs(ys[1] - ys[0]) / 2
    west, south = utm_to_lonlat(xs.min() - hx, ys.min() - hy, crs)
    east, north = utm_to_lonlat(xs.max() + hx, ys.max() + hy, crs)
    if background is not None:
        import matplotlib
        rgba = matplotlib.colormaps["RdYlGn"](np.clip(np.nan_to_num(background, nan=0) / 0.9, 0, 1))
        rgba[..., 3] = np.where(np.isfinite(background), 1.0, 0.0)
        ImageOverlay(image=rgba, bounds=[[float(south), float(west)], [float(north), float(east)]], opacity=0.45,
                     name="Median NDVI (MODIS, approximate position)", show=False, mercator_project=True).add_to(m)
    folium.Rectangle([[float(south), float(west)], [float(north), float(east)]], color="white", weight=1.5, fill=False,
                     tooltip="Window of the cube").add_to(m)

    for name, (row, col) in pixels.items():
        colour = colours.get(name, "#FFD400")
        lon, lat = pixel_to_lonlat(cube, row, col, crs=crs)
        label = f"{name}: row {row}, col {col} | lat {lat:.5f}, lon {lon:.5f}"
        folium.Polygon(pixel_footprint(cube, row, col, crs=crs), color=colour, weight=3, fill=False, tooltip=label).add_to(m)
        folium.CircleMarker([lat, lon], radius=4, color=colour, fill=True, fill_opacity=1, tooltip=label).add_to(m)
    m.add_child(folium.LatLngPopup())                              # click -> shows latitude and longitude
    folium.LayerControl(collapsed=False).add_to(m)
    return m
