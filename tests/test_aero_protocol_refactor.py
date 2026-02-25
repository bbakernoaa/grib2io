
import sys
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr
import pandas as pd
import pytest

# Mock grib2io
mock_grib2io = MagicMock()
sys.modules['grib2io'] = mock_grib2io
sys.modules['grib2io._grib2io'] = MagicMock()
sys.modules['grib2io.templates'] = MagicMock()
sys.modules['grib2io.tables'] = MagicMock()
sys.modules['grib2io.utils'] = MagicMock()

import grib2io.xarray_backend as xarray_backend
from grib2io.xarray_backend import parse_data_model

def test_parse_data_model_lazy_logic():
    # This test verifies that parse_data_model handles both numpy and dask data identically

    # Mocking decode function used in parse_data_model
    def mock_decode(values, table):
        return values.astype(str) # Simple mock

    with patch('grib2io.xarray_backend._decode_code', side_effect=mock_decode):
        # 1. Eager version
        ds_eager = xr.Dataset(
            {"TMP": (("y", "x"), np.array([[1, 2], [3, 4]], dtype=np.float32))},
            coords={"typeOfAerosol": ((), 62001)} # Use a dummy value
        )
        ds_eager["TMP"].attrs = {
            "typeOfFirstFixedSurface": ("Ground or Water Surface", "unknown"),
            "typeOfAerosol": 62001
        }

        ds_eager_parsed = parse_data_model(ds_eager, "nws-viz")

        # 2. Lazy version
        try:
            import dask.array as da
            ds_lazy = xr.Dataset(
                {"TMP": (("y", "x"), da.from_array(np.array([[1, 2], [3, 4]], dtype=np.float32), chunks=(1, 2)))},
                coords={"typeOfAerosol": ((), 62001)}
            )
            ds_lazy["TMP"].attrs = ds_eager["TMP"].attrs.copy()

            ds_lazy_parsed = parse_data_model(ds_lazy, "nws-viz")

            # Assertions
            xr.testing.assert_allclose(ds_eager_parsed.tmp, ds_lazy_parsed.tmp.compute())
            assert "aerosol_type" in ds_eager_parsed.coords
            assert "aerosol_type" in ds_lazy_parsed.coords
            assert ds_lazy_parsed.aerosol_type.chunks is not None # Should be lazy!

        except ImportError:
            print("Dask not installed, skipping lazy check")

if __name__ == "__main__":
    test_parse_data_model_lazy_logic()
    print("Double-check test logic verified!")
