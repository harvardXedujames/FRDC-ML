# preprocessing.scale

<tldr>
Scales the NDArray bands.
</tldr>

## Functions

<warning>
Scale assumes H x W x C, where C is the number of bands.
</warning>

<deflist type="medium">
<def title="scale_0_1_per_band">
<b>Scales the NDArray bands to [0, 1] per band.</b><br/>
</def>
<def title="scale_normal_per_band">
<b>Scales the NDArray bands to zero mean unit variance per band.</b><br/>
</def>
<def title="scale_static_per_band">
<b>Scales the NDArray bands by a predefined configuration.</b><br/>
</def>
</deflist>

## Usage

```python
from frdc.load import FRDCDataset
from frdc.preprocess.scale import (
    scale_0_1_per_band, scale_normal_per_band, scale_static_per_band
)
from frdc.conf import BAND_MAX_CONFIG

ds = FRDCDataset(site='chestnut_nature_park',
                 date='20201218',
                 version=None, )
ar, order = ds.get_ar_bands()
ar_01 = scale_0_1_per_band(ar)
ar_norm = scale_normal_per_band(ar)
ar_static = scale_static_per_band(ar, order, BAND_MAX_CONFIG)
```

> The static scaling has a default config infers the max bit depth from the
> capturing device.

## API

<deflist>
<def title="scale_0_1_per_band(ar)">
<b>Scales the NDArray bands to [0, 1] per band.</b><br/>
<code-block lang="tex">
(x - \min(x)) / (\max(x) - \min(x))
</code-block>
</def>
<def title="scale_normal_per_band(ar)">
<b>Scales the NDArray bands to zero mean unit variance per band.</b><br/>
<code-block lang="tex">
(x - \mu) / \sigma
</code-block>
</def>
<def title="scale_static_per_band(ar, order, config)">
<b>Scales the NDArray bands by a predefined configuration.</b><br/>
The <code>config</code> is of <code>dict[str, tuple[int, int]]</code> where
the key is the band name, and the value is a tuple of <code>(min, max)</code>.
Take a look at <code>frdc.conf.BAND_MAX_CONFIG</code> for an example.
<code-block lang="tex">
(x - c_0) / (c_1 - c_0)
</code-block>
</def>
</deflist>