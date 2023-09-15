# FRDC Resources

We have 2 type of resources:
1) Raw: Uncompressed, raw data from our UAV drones
2) Debug: Compressed version of the raw data used for
   1) Unit Testing
   2) Integration Testing
   3) Experimentation Debugging

The differences are:
1) Debug is stored lossy, in `.jpeg` format. While Raw is in lossless `.tiff`
2) Debug is committed to Git, while Raw only on demand.
3) Debug is used in our tests, while Raw only if absolutely necessary. This is to reduce I/O costs.

## Structure
The file structure should follow this

Each folder with data should have the following files:
- `result.tif`
- `result_Blue.tif`
- `result_Green.tif`
- `result_NIR.tif`
- `result_Red.tif`
- `result_RedEdge.tif`
- `bounds.csv` _(Optional: Will eventually be deprecated)_
```
surveyed-site/
    survey-date/
        result.tif
        result_Blue.tif
        result_Green.tif
        result_NIR.tif
        result_Red.tif
        result_RedEdge.tif
        bounds.csv
```