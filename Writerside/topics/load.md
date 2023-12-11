# load

> You need to Set-Up [Google Cloud](Getting-Started.md#gcloud) with the
> appropriate permissions to use this library.
> {style='warning'}

<tldr>
Load datasets from our GCS bucket.
</tldr>

## Classes

<deflist>
<def title="FRDCDataset">
Uses GCS utils to download and load the dataset.
It also implements useful helper functions to load FRDC-specific datasets,
such as loading our images and labels.
</def>
<def title="GCSConfig">
A configuration object that controls how downloads are done.
See Functions below
</def>
</deflist>

## Functions

These are defined in the top-level load.gcs module.

<deflist>
<def title="download">
Downloads a file from Google Cloud Storage. If the file already
exists locally, and the hashes match, it will not download the file
</def>
<def title="list_gcs_datasets">
Lists all datasets in the bucket as a DataFrame.
</def>
<def title="open_file">
Opens a file from Google Cloud Storage. If the file already
exists locally, and the hashes match, it will not download the file
</def>
<def title="open_image">
Opens an image from Google Cloud Storage as a PIL Image.
If the file already exists locally, and the hashes match, it will not download the file
</def>
</deflist>

## Usage

An example loading our Chestnut Nature Park dataset. We retrieve the

- hyperspectral bands
- order of the bands
- bounding boxes
- labels

```python
from frdc.load import FRDCDataset

ds = FRDCDataset(site='chestnut_nature_park',
                 date='20201218',
                 version=None)
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
```

### Custom Authentication & Downloads {collapsible="true"}

If you need granular control over

- where the files are downloaded
- the credentials used
- the project used
- the bucket used

Then pass in a `GCSConfig` object to `FRDCDataset`.

```python
from frdc.load import FRDCDataset, GCSConfig

cfg = GCSConfig(credentials=...,
               local_dir=...,
               project_id=...,
               bucket_name=...,
               local_exists_ok=True)
ds = FRDCDataset(site='chestnut_nature_park',
                 date='20201218',
                 version=None, )
ar, order = ds.get_ar_bands(dl_config=cfg)
bounds, labels = ds.get_bounds_and_labels(dl_config=cfg)
```

If you have a file not easily downloadable by `FRDCDataset`, use the `gcs`
module utility functions

```python
from frdc.load import download

download(fp="path/to/gcs/file")
```

<tip>This will automatically save the file to the local dataset root dir.</tip>

## API

### FRDCDataset

<deflist>
<def title="FRDCDataset(site, date, version, dl)">
<b>Initializes the dataset.</b><br/>
This doesn't immediately download the dataset, but only when you call the
<code>get_*</code> functions.<br/>
The site, date, version must match the dataset path on GCS. For example
if the dataset is at
<code>gs://frdc-scan/my-site/20201218/90deg/map</code>,
<list>
<li><code>site='my-site'</code></li>
<li><code>date='20201218'</code></li>
<li><code>version='90deg/map'</code></li>
</list>
If the dataset doesn't have a "version", for example:
<code>gs://frdc-scan/my-site/20201218</code>,
then you can pass in <code>version=None</code>.<br/>
<note>
If you don't want to search up GCS, you can use <code>gcs.utils</code>
to list all datasets, and their versions with 
<code>list_gcs_datasets()</code>
</note>
</def>
<def title="get_ar_bands()">
<b>Gets the NDArray bands (H x W x C) and channel order as 
<code>tuple[np.ndarray, list[str]]</code>.</b><br/>
This downloads (if missing) and retrieves the stacked NDArray bands.
This wraps around <code>get_ar_bands_as_dict()</code>, thus if you want more
control over how the bands are loaded, use that instead. 
</def>
<def title="get_ar_bands_as_dict()">
<b>Gets the NDArray bands (H x W) as a <code>dict[str, np.ndarray]</code>.</b><br/>
This downloads (if missing) and retrieves the individual NDArray bands as a
dictionary. The keys are the band names, and the values are the NDArray bands.
</def>
<def title="get_bounds_and_labels()">
<b>Gets the bounding boxes and labels as 
<code>tuple[list[Rect], list[str]]</code>.</b><br/>
This downloads (if missing) and retrieves the bounding boxes and labels as a
tuple. The first element is a list of bounding boxes, and the second element
is a list of labels.<br/>   
<tip>The bounding boxes are in the format of <code>Rect</code>, a 
<code>namedtuple</code> of x0, y0, x1, y1.</tip>
</def>
</deflist>

### load.gcs

<deflist>
<def title="list_gcs_datasets(anchor)">
<b>Lists all GCS datasets in the bucket as <code>DataFrame</code></b><br/>
This works by checking which folders have a specific file, which we call the
<code>anchor</code>.
</def>
<def title="download(fp, config)">
<b>Downloads a file from GCS.</b><br/>
Takes in a path glob <code>fp</code>,
a string containing wildcards, and downloads exactly 1 file.
If it matches 0 or more than 1 file, it will raise an error.<br/>
It uses the configuration from <code>config</code>, which controls the behavior
of how downloads are done. The default is usually fine. <br/>
The download will skip if the file exists and the hash matches.
</def>
<def title="open_file(fp, config)">
<b>Opens a file from GCS.</b><br/>
Wraps around <code>download()</code>, but returns a file handle instead.
</def>
<def title="open_image(fp, config)">
<b>Opens an image from GCS as a PIL Image.</b><br/>
Wraps around <code>download()</code>, but returns a PIL Image instead.
</def>
</deflist>
