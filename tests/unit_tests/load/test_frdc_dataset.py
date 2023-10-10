from frdc.conf import BAND_CONFIG
from frdc.utils import Rect


def test_get_ar_bands_as_dict(ds):
    d = ds.get_ar_bands_as_dict(BAND_CONFIG)
    assert set(d.keys()) == set(d.keys())


def test_get_ar_bands(ar, order):
    assert ar.shape[-1] == len(order)


def test_get_bounds(ds):
    bounds, labels = ds.get_bounds_and_labels()
    assert all([isinstance(b, Rect) for b in bounds])
    assert len(bounds) == len(labels)
