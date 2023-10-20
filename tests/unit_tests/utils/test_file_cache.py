from pathlib import Path

import numpy as np
import pytest

from frdc.utils.file_cache import file_cache


@pytest.mark.cupy
def test_caching():
    """This test ensures that the file_cache decorator works as expected.

    1) It should cache the result of the function call.
    2) It should not call the function again if the result is cached.
    3) It should call the function again if the result is not cached.

    """
    # Prepare the cache directory
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    for f in cache_dir.glob("*.npy"):
        f.unlink()

    # This keeps track of how many times the function is called.
    calls = 0

    @file_cache(
        fn_cache_fp=lambda x: cache_dir / f"{x}.npy",
        fn_save_object=np.save,
        fn_load_object=np.load,
    )
    def my_fn(x):
        # Increments calls for each call to this function.
        nonlocal calls
        calls += 1

        np.random.seed(x)
        return np.random.randint(0, 10, [5, 5])

    # First call should call the function.
    my_fn(1)
    assert calls == 1
    # Second call with the same argument should not call the function.
    my_fn(1)
    assert calls == 1
    # Third call with a different argument should call the function.
    my_fn(2)
    assert calls == 2

    # Clean up
    for f in cache_dir.glob("*.npy"):
        f.unlink()
