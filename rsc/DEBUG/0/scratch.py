import numpy as np
from PIL import Image

# %%
for i in (
    "result.tif",
    "result_Red.tif",
    "result_Green.tif",
    "result_Blue.tif",
    "result_RedEdge.tif",
    "result_NIR.tif",
):
    im = Image.open(i)
    ar = np.array(im)
    ar_crop = ar[2000:5000:20, 1000:3000:20]
    im = Image.fromarray(ar_crop)
    im.save(i)
