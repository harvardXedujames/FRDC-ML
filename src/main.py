""" Entry point for the whole ML pipeline. """
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure

from frdc.load import FRDCDownloader
from frdc.preprocess.preprocess import extract_segments, compute_segments_mask, scale_0_1_per_band

# %%
ar = FRDCDownloader().load_dataset(site='chestnut_nature_park', date='20201218', version=None)
# %%
ar_segments_mask = compute_segments_mask(
    ar, peaks_footprint=200, min_crown_hole=3000, min_crown_size=3000,
    connectivity=1, watershed_compactness=10,
)
plt.imshow(ar_segments_mask)
plt.show()
# %%
ar_segments_mask.max()
# %%
ar_segments = extract_segments(scale_0_1_per_band(ar), ar_segments_mask)
# %%
ar_segments = sorted(ar_segments, key=lambda x: x.size, reverse=True)
#%%

plt.imshow(ar_segments[-1][:, :, [2, 1, 0]])
plt.show()
# %%
segment = ar_segments_mask == 2
coords = np.argwhere(segment)
x0, y0 = coords.min(axis=0)
x1, y1 = coords.max(axis=0) + 1

# %%

# %%
labels = measure.label(ar_segments_mask == 2)
regions = measure.regionprops(labels)
regions[0].bbox
