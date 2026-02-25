
import sys
from unittest.mock import MagicMock, patch

# Create a mock for grib2io and its submodules
mock_grib2io = MagicMock()
mock_grib2io._grib2io = MagicMock()
mock_grib2io.templates = MagicMock()
mock_grib2io.tables = MagicMock()
mock_grib2io.utils = MagicMock()

# Ensure get_table returns a dict
mock_grib2io.tables.get_table.return_value = {}
mock_grib2io.tables.get_value_from_table.return_value = "mock_value"

# Set up the sys.modules
sys.modules['grib2io'] = mock_grib2io
sys.modules['grib2io._grib2io'] = mock_grib2io._grib2io
sys.modules['grib2io.templates'] = mock_grib2io.templates
sys.modules['grib2io.tables'] = mock_grib2io.tables
sys.modules['grib2io.utils'] = mock_grib2io.utils

import numpy as np
import xarray as xr
import pandas as pd
import pytest
import datetime

# Now import the backend
import grib2io.xarray_backend as xarray_backend
from grib2io.xarray_backend import parse_data_model, open_mfdataset, _open_dataset_from_index

def test_parse_data_model_aero_eager():
    # Create a mock dataset with aerosol attributes
    ds = xr.Dataset(
        {"TMP": (("y", "x"), np.random.rand(10, 10).astype(np.float32))},
        coords={
            "typeOfAerosol": ((), 1),
            "scaledValueOfFirstWavelength": ((), 550),
            "refDate": ((), pd.Timestamp("2023-01-01")),
            "leadTime": ((), pd.Timedelta("1h")),
        }
    )
    # Mock shortname_to_cf table
    xarray_backend.tables.get_table.side_effect = lambda x: {} if x == "shortname_to_cf" else {}

    ds["TMP"].attrs = {
        "units": "K",
        "typeOfFirstFixedSurface": ("Ground or Water Surface", "unknown"),
        "typeOfAerosol": 1,
        "scaledValueOfFirstWavelength": 550
    }

    # Mock _decode_ptype and _decode_code if needed
    with patch('grib2io.xarray_backend._decode_code', return_value=np.array(["AerosolType"])):
        parsed_ds = parse_data_model(ds, "nws-viz")

        assert "aerosol_type" in parsed_ds.coords
        assert "scaled_first_wavelength" in parsed_ds.coords
        assert parsed_ds.scaled_first_wavelength.item() == 550
        assert "history" in parsed_ds.attrs

def test_parse_data_model_aero_lazy():
    # Use dask
    try:
        import dask.array as da
    except ImportError:
        print("Dask not installed, skipping lazy test")
        return

    ds = xr.Dataset(
        {"TMP": (("y", "x"), da.random.random((10, 10), chunks=(5, 5)).astype(np.float32))},
        coords={
            "typeOfAerosol": ((), 1),
            "scaledValueOfFirstWavelength": ((), 550),
            "refDate": ((), pd.Timestamp("2023-01-01")),
            "leadTime": ((), pd.Timedelta("1h")),
        }
    )
    ds["TMP"].attrs = {
        "units": "K",
        "typeOfFirstFixedSurface": ("Ground or Water Surface", "unknown"),
        "typeOfAerosol": 1
    }

    with patch('grib2io.xarray_backend._decode_code', return_value=np.array(["AerosolType"])):
        parsed_ds = parse_data_model(ds, "nws-viz")

        assert "aerosol_type" in parsed_ds.coords
        assert "scaled_first_wavelength" in parsed_ds.coords
        # Check if it is still dask-backed
        assert hasattr(parsed_ds.TMP.data, "chunks")

def test_time_axis_validDate_logic():
    # Mock Grib2Message
    msg = MagicMock()
    msg.validDate = datetime.datetime(2023, 1, 1, 1, 0, 0)
    msg.refDate = datetime.datetime(2023, 1, 1, 0, 0, 0)
    msg.leadTime = datetime.timedelta(hours=1)
    msg.shortName = "TMP"
    msg.nx = 10
    msg.ny = 10
    msg.section0 = [0,0,0,2,1000]
    msg.section1 = [0]*13
    msg.section2 = b''
    msg.section3 = [0, 100, 0, 0, 0] + [0]*19
    msg.section4 = [0, 0] + [0]*15
    msg.section5 = [0, 0] + [0]*5
    msg.typeOfValues = 0
    msg.fullName = "Temperature"
    msg.units = "K"
    msg.originatingCenter.definition = "NCEP"
    msg.originatingSubCenter.definition = "NCEP"
    msg.masterTableInfo.definition = "8"
    msg.productDefinitionTemplateNumber.value = 0
    msg.latlons.return_value = (np.zeros((10,10)), np.zeros((10,10)))

    # Mock index
    index = pd.DataFrame({
        "msg": [msg],
        "shortName": ["TMP"],
        "nx": [10],
        "ny": [10],
        "sectionOffset": [[0]*8],
    })

    # Mock grib2io.msgs_from_index
    with patch('grib2io.xarray_backend.msgs_from_index', return_value=[msg]):
        # Mock build_da_without_coords to return a mock DataArray
        da = xr.DataArray(np.zeros((1, 10, 10)), dims=['validDate', 'y', 'x'], name="TMP")
        da.attrs = {"units": "K"}
        with patch('grib2io.xarray_backend.build_da_without_coords', return_value=da):
            # Call _open_dataset_from_index
            ds = _open_dataset_from_index(index, "dummy.grib2", time_axis='validDate')

            assert 'validDate' in ds.dims
            assert ds.validDate.values[0] == msg.validDate

if __name__ == "__main__":
    test_parse_data_model_aero_eager()
    print("Eager test passed")
    test_parse_data_model_aero_lazy()
    print("Lazy test passed")
    test_time_axis_validDate_logic()
    print("Time axis test passed")
