# Dataset

> You need to Set-Up [Google Cloud](Getting-Started.md#gcloud) with the
> appropriate permissions to use this library.
> {style='warning'}

<tldr>
Load datasets from our GCS bucket.
</tldr>

## Classes

<deflist>
<def title="FRDCDownloader">
This facilitates authentication and downloading from GCS.
</def>
<def title="FRDCDataset">
This uses the Downloader to download and load the dataset.
It also implements useful helper functions to load FRDC-specific datasets,
such as loading our images and labels.
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
                 version=None, )
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
```

### Custom Authentication & Downloads {collapsible="true"}

If you need granular control over

- where the files are downloaded
- the credentials used
- the project used
- the bucket used

Then pass in a `FRDCDownloader` object to `FRDCDataset`.

```python
from frdc.load import FRDCDownloader, FRDCDataset

dl = FRDCDownloader(credentials=...,
                    local_dataset_root_dir=...,
                    project_id=...,
                    bucket_name=...)
ds = FRDCDataset(site='chestnut_nature_park',
                 date='20201218',
                 version=None,
                 dl=dl)
ar, order = ds.get_ar_bands()
bounds, labels = ds.get_bounds_and_labels()
```

If you have a file not easily downloadable by `FRDCDataset`, you can use
`FRDCDownloader` to download it.

```python
from frdc.load import FRDCDownloader

dl = FRDCDownloader(credentials=...,
                    local_dataset_root_dir=...,
                    project_id=...,
                    bucket_name=...)

dl.download_file(path_glob="path/to/gcs/file")
```

<tip>This will automatically save the file to the local dataset root dir.</tip>

## API

### FRDCDataset

<deflist>
<def title="FRDCDataset(site, date, version, dl)">
<b>Initializes the dataset downloader.</b><br/>
This doesn't immediately download the dataset, but only when you call the
<code>get_*</code> functions.<br/>
The site, date, version must match the dataset path on GCS. For example
if the dataset is at
<code>gs://frdc-scan/my-site/date/90deg/map</code>,
<list>
<li>site: <code>'my-site'</code></li>
<li>date: <code>'20201218'</code></li>
<li>version: <code>'90deg/map'</code></li>
</list>
<note>
If you don't want to search up GCS, you can use FRDCDownloader to list all
datasets, and their versions with 
<code>FRDCDownloader().list_gcs_datasets()</code>
</note>
<tip>
If <code>dl</code> is None, it will create a new FRDCDownloader. Usually,
you don't need to pass this in unless you have a custom credential, or project.
</tip>
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

### FRDCDownloader

<deflist>
<def title="list_gcs_datasets(anchor)">
<b>Lists all GCS datasets in the bucket as a DataFrame.</b><br/>
This works by checking which folders have a specific file, which we call the
<code>anchor</code>.
</def>
<def title="download_file(path_glob, local_exists_ok)">
<b>Downloads a file from GCS.</b><br/>
This takes in a path glob, a string containing wildcards, and downloads exactly
1 file. If it matches 0 or more than 1 file, it will raise an error.<br/>
If <code>local_exists_ok</code> is True, it will not download the file if it
already exists locally. However, if it's False, it will download the file
only if the hashes don't match.

<note>Use this if you have a file on GCS that can't be downloaded via
FRDCDataset.</note>
</def>
</deflist>