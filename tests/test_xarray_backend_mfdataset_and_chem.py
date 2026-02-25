import pytest
import xarray as xr
import numpy as np
import pandas as pd
from grib2io import xarray_backend
from unittest.mock import MagicMock, patch

def test_parse_data_model_chemical_constituents():
    """
    Test that parse_data_model correctly handles chemical constituent coordinates.

    This test verifies that coordinates like typeOfAerosol, constituentType,
    and sourceSinkIndicator are correctly renamed to snake_case and decoded
    to their human-readable definitions using xr.apply_ufunc.
    """
    # Create a dummy dataset with chemical coordinates
    ds = xr.Dataset(
        data_vars={
            "O3MR": (("level", "y", "x"), np.ones((1, 2, 2), dtype="float32")),
        },
        coords={
            "level": [1],
            "y": [0, 1],
            "x": [0, 1],
            "typeOfAerosol": (("level",), [1]),
            "constituentType": (("level",), [0]),
            "firstWavelength": (("level",), [550.0]),
            "sourceSinkIndicator": (("level",), [1]),
        },
    )
    # Add attributes required by nws-viz
    ds.O3MR.attrs = {
        "units": "kg kg-1",
        "typeOfFirstFixedSurface": ("Isobaric Surface", "Pa"),
        "GRIB2IO_section0": [0,0,0,0,0],
        "GRIB2IO_section1": [0,0,0,0,0,0,0,0,0,0,0,0,0],
        "GRIB2IO_section3": [0,0,0,0,0,0,0,0,0,0,0,0,0],
        "GRIB2IO_section4": [0,0,0,0,0,0,0,0,0,0,0,0,0],
        "GRIB2IO_section5": [0,0,0,0,0,0,0,0,0,0,0,0,0],
        "fullName": "Ozone Mixing Ratio",
        "shortName": "O3MR",
    }

    # Mock tables.get_value_from_table and get_table
    with patch("grib2io.tables.get_value_from_table") as mock_get_value, \
         patch("grib2io.tables.get_table") as mock_get_table:

        def side_effect_val(val, table):
            if table == "4.233":
                return ["Dust", "DUST"]
            if table == "4.230":
                return ["Ozone", "O3"]
            if table == "4.238":
                return ["Sink", "SINK"]
            if table == "4.201":
                return ["Rain", "RAIN"]
            if table == "4.91":
                return ["Smaller than first limit", "LT1"]
            return val
        mock_get_value.side_effect = side_effect_val

        mock_get_table.return_value = {} # for shortname_to_cf

        # Test nws-viz data model
        ds_nws = xarray_backend.parse_data_model(ds, "nws-viz")

        # Check renames
        assert "aerosol_type" in ds_nws.coords
        assert "constituent_type" in ds_nws.coords
        assert "first_wavelength" in ds_nws.coords
        assert "source_sink_indicator" in ds_nws.coords

        # Check decodings
        assert ds_nws.aerosol_type.values[0] == "Dust"
        assert ds_nws.constituent_type.values[0] == "Ozone"
        assert ds_nws.source_sink_indicator.values[0] == "Sink"

        # Verify with Dask-backed data
        ds_lazy = ds.chunk({"y": 1})
        ds_nws_lazy = xarray_backend.parse_data_model(ds_lazy, "nws-viz")

        # Check that it's still lazy
        assert hasattr(ds_nws_lazy.aerosol_type.data, "dask")

        # Check decodings (this triggers compute)
        assert ds_nws_lazy.aerosol_type.values[0] == "Dust"
        assert ds_nws_lazy.constituent_type.values[0] == "Ozone"
        assert ds_nws_lazy.source_sink_indicator.values[0] == "Sink"

def test_open_mfdataset_preprocess():
    """
    Test that open_mfdataset correctly applies the preprocess function.

    This test mocks the internal index gathering and dataset creation to
    verify that the provided preprocess function is called on each individual
    file's dataset before they are combined.
    """
    with patch("grib2io.open") as mock_grib_open, \
         patch("grib2io.xarray_backend.msgs_from_index") as mock_msgs, \
         patch("grib2io.xarray_backend._open_dataset_from_index") as mock_open_from_idx:

        # Setup mock grib file object
        mock_f = MagicMock()
        mock_f._index = {"ny": [2], "nx": [2]}
        mock_grib_open.return_value.__enter__.return_value = mock_f

        mock_msgs.return_value = [MagicMock()]

        # Mock the final dataset
        ds_mock = xr.Dataset({"TMP": (("y", "x"), np.ones((2, 2)))})
        mock_open_from_idx.return_value = ds_mock

        # Preprocess function
        def preprocess(ds):
            ds.attrs["preprocessed"] = True
            return ds

        # Call open_mfdataset
        ds = xarray_backend.open_mfdataset(["file1.grib2"], preprocess=preprocess)

        # Check if preprocess was called
        assert ds.attrs["preprocessed"] is True

        # Verify that _open_dataset_from_index was called with a list of filenames
        args, kwargs = mock_open_from_idx.call_args
        assert isinstance(args[1], list)
        assert args[1] == ["file1.grib2"]

def test_make_variables_optimization():
    """
    Test the optimization in make_variables using groupby.

    This test verifies that make_variables correctly identifies and separates
    multiple GRIB2 variables from a single index DataFrame and creates the
    appropriate frames and cubes for each.
    """
    # Verify that make_variables correctly handles multiple variables
    index = pd.DataFrame({
        "shortName": ["TMP", "TMP", "HGT", "HGT"],
        "ny": [2, 2, 2, 2],
        "nx": [2, 2, 2, 2],
        "leadTime": [pd.Timedelta(hours=1), pd.Timedelta(hours=2), pd.Timedelta(hours=1), pd.Timedelta(hours=2)],
    })
    # Mock Grib2Message
    index["msg"] = [MagicMock() for _ in range(4)]
    for msg in index["msg"]:
        msg.latlons.return_value = (np.zeros((2, 2)), np.zeros((2, 2)))

    non_geo_dims = {"leadTime": ["leadTime"]}

    frames, cubes, extra_geo = xarray_backend.make_variables(index, "dummy.grib2", non_geo_dims)

    assert len(frames) == 2
    assert len(cubes) == 2
    # The order depends on groupby sorting, which is level=0 (shortName)
    shortnames = sorted([f.shortName.iloc[0] for f in frames])
    assert shortnames == ["HGT", "TMP"]

if __name__ == "__main__":
    pytest.main([__file__])
