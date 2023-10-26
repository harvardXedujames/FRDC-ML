# preprocessing.glcm_padded

<tldr>
Computes the GLCM of the NDArray bands with padding.
</tldr>

> This is largely a handy wrapper around
> my [glcm-cupy](https://github.com/Eve-ning/glcm-cupy) package.
> This auto-computes the necessary padding so that the GLCM is the same size
> as the original image.

> The GLCM computation is rather slow, so it is recommended to use it
> only if necessary.
> {style='warning'}

## Functions

<warning>
Assumes shape H x W x C, where C is the number of bands.
</warning>

<deflist type="medium">
<def title="glcm_padded">
<b>Computes the GLCM of the NDArray bands with padding.</b>
</def>
<def title="glcm_padded_cached">
<b>Computes the GLCM of the NDArray bands with padding, and caches it.</b>
</def>
<def title="append_glcm_padded_cached">
<b>Computes the GLCM of the NDArray bands with padding, and caches it and
also appends it onto the original array.</b>
</def>
</deflist>

## Usage

We show a few examples of how to use the GLCM functions.

```python
import numpy as np
from glcm_cupy import Features

from frdc.preprocess.glcm_padded import (
    append_glcm_padded_cached, glcm_padded, glcm_padded_cached
)

ar = np.random.rand(50, 25, 4)

# Returns a shape of H x W x C x GLCM Features
ar_glcm = glcm_padded(ar, bin_from=1, bin_to=4, radius=3, )

# Returns a shape of H x W x C x 2
ar_glcm_2_features = glcm_padded(ar, bin_from=1, bin_to=4, radius=3,
                                 features=[Features.CONTRAST,
                                           Features.CORRELATION])

# Returns a shape of H x W x C x GLCM Features
ar_glcm_cached = glcm_padded_cached(ar, bin_from=1, bin_to=4, radius=3)

# Returns a shape of H x W x (C x GLCM Features + C)
ar_glcm_cached_appended = append_glcm_padded_cached(ar, bin_from=1, bin_to=4,
                                                    radius=3)

```

- `ar_glcm` is the GLCM of the original array, with the last dimension being
  the GLCM features. The number of features is determined by the `features`
  parameter, which defaults to all features.
- `ar_glcm_2_features` selects only 2 features, with the last dimension being
  the 2 GLCM features specified.
- `ar_glcm_cached` caches the GLCM so that if you call it again,
  it will return the cached version. It stores its data at the project root
  dir, under `.cache/`.
- `ar_glcm_cached_appended` is a wrapper around `ar_glcm_cached`, it
  appends the GLCM features onto the original array. It's equivalent to calling
  `ar_glcm_cached` and then `np.concatenate` on the final axes.

### Caching

GLCM is an expensive operation, thus we recommend to cache it if the input
parameters will be the same. This is especially useful if you're
experimenting with the same dataset with constant parameters.

> This cache is automatically invalidated if the parameters change. Thus, if
> you perform augmentation, the cache will not be used and will be recomputed.
> This can be wasteful, so it is recommended to perform augmentation after
> the GLCM computation if possible.
> {style='warning'}

> The cache is stored at the project root dir, under `.cache/`. It is safe to
> delete this folder if you want to clear the cache.
> {style='note'}

## API

<deflist>
<def title="glcm_padded(ar, bin_from, bin_to, radius, step_size, features)">
<b>Computes the GLCM of the NDArray bands with padding.</b><br/>
<list>
<li><code>ar</code> is the input array</li>
<li><code>bin_from</code> is the upper bound of the input</li>
<li><code>bin_to</code> is the upper bound of the GLCM input, i.e. the 
resolution that GLCM operates on</li>
<li><code>radius</code> is the radius of the GLCM</li>
<li><code>step_size</code> is the step size of the GLCM</li>
<li><code>features</code> is the list of GLCM features to compute</li>
</list>
The return shape is
<code-block lang="tex">
H \times W \times C \times \text{GLCM Features}
</code-block>
See <code>glcm_cupy</code> for the GLCM Features.
</def>
<def title="glcm_padded_cached(ar, bin_from, bin_to, radius, step_size, features)">
<b>Computes the GLCM of the NDArray bands with padding, and caches it.</b><br/>
See <code>glcm_padded</code> for the parameters and output shape
</def>
<def title="append_glcm_padded_cached(ar, bin_from, bin_to, radius, step_size, features)">
<b>Computes the GLCM of the NDArray bands with padding, and caches it and
also appends it onto the original array.</b><br/>
See <code>glcm_padded</code> for the parameters<br/>
The return shape is:
<code-block lang="tex">
H \times W \times (C \times \text{GLCM Features} + C)
</code-block>
The function automatically flattens the last 2 dimensions of the GLCM
features, and appends it onto the original array.
</def>
</deflist>
