import sys
from unittest.mock import MagicMock, patch

# Mock g2clib and other compiled extensions before importing grib2io
mock_g2clib = MagicMock()
mock_g2clib.__version__ = '1.0.0'
mock_g2clib._has_jpeg = 1
mock_g2clib._has_png = 1
mock_g2clib._has_aec = 1
sys.modules['grib2io.g2clib'] = mock_g2clib
mock_iplib = MagicMock()
sys.modules['grib2io.iplib'] = mock_iplib
mock_redtoreg = MagicMock()
sys.modules['grib2io.redtoreg'] = mock_redtoreg

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from grib2io import xarray_backend

# Check for NumPy 2.0+ StringDType
_HAS_STRINGDTYPE = hasattr(np, 'dtypes') and hasattr(np.dtypes, 'StringDType')


@pytest.fixture
def mock_grib2io_backend():
    with patch('grib2io.tables.get_value_from_table') as mock_lookup, patch(
        'grib2io.tables.get_table'
    ) as mock_get_table:
        mock_lookup.side_effect = lambda val, tbl: f'PTYPE_{val}'
        mock_get_table.return_value = {}
        yield mock_lookup, mock_get_table


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
            'O3MR': (('level', 'y', 'x'), np.ones((1, 2, 2), dtype='float32')),
            'AOTK': (('level', 'y', 'x'), np.ones((1, 2, 2), dtype='float32')),
        },
        coords={
            'level': [1],
            'y': [0, 1],
            'x': [0, 1],
            'typeOfAerosol': (('level',), [1]),
            'constituentType': (('level',), [0]),
            'firstWavelength': (('level',), [550.0]),
            'sourceSinkIndicator': (('level',), [1]),
            'typeOfIntervalForAerosolSize': (('level',), [1]),
        },
    )
    # Add attributes required by nws-viz
    ds.O3MR.attrs = {
        'units': 'kg kg-1',
        'typeOfFirstFixedSurface': ('Isobaric Surface', 'Pa'),
        'GRIB2IO_section0': [0, 0, 0, 0, 0],
        'GRIB2IO_section1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GRIB2IO_section3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GRIB2IO_section4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GRIB2IO_section5': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'fullName': 'Ozone Mixing Ratio',
        'shortName': 'O3MR',
    }

    # Mock tables.get_value_from_table and get_table
    with patch('grib2io.tables.get_value_from_table') as mock_get_value, patch(
        'grib2io.tables.get_table'
    ) as mock_get_table:

        def side_effect_val(val, table):
            if table == '4.233':
                return ['Dust', 'DUST']
            if table == '4.230':
                return ['Ozone', 'O3']
            if table == '4.238':
                return ['Sink', 'SINK']
            if table == '4.201':
                return ['Rain', 'RAIN']
            if table == '4.91':
                return ['Smaller than first limit', 'LT1']
            return val

        mock_get_value.side_effect = side_effect_val

        mock_get_table.return_value = {
            'O3MR': {
                'cf_standard_name': 'mass_fraction_of_ozone_in_air',
                'cf_cell_methods': None,
            },
            'AOTK': {
                'cf_standard_name': 'atmosphere_optical_thickness_due_to_ambient_aerosol',
                'cf_cell_methods': None,
            },
        }

        # Test nws-viz data model
        ds_nws = xarray_backend.parse_data_model(ds, 'nws-viz')

        # Check renames
        assert 'aerosol_type' in ds_nws.coords
        assert 'constituent_type' in ds_nws.coords
        assert 'first_wavelength' in ds_nws.coords
        assert 'source_sink_indicator' in ds_nws.coords
        assert 'aerosol_size_interval_type' in ds_nws.coords

        # Check decodings
        assert ds_nws.aerosol_type.values[0] == 'Dust'
        assert ds_nws.constituent_type.values[0] == 'Ozone'
        assert ds_nws.source_sink_indicator.values[0] == 'Sink'
        assert ds_nws.aerosol_size_interval_type.values[0] == 'Smaller than first limit'

        # Check CF standard names
        assert ds_nws.o3mr.attrs['standard_name'] == 'mass_fraction_of_ozone_in_air'
        assert (
            ds_nws.aotk.attrs['standard_name']
            == 'atmosphere_optical_thickness_due_to_ambient_aerosol'
        )

        # Verify with Dask-backed data
        ds_lazy = ds.chunk({'y': 1})
        ds_nws_lazy = xarray_backend.parse_data_model(ds_lazy, 'nws-viz')

        # Check that it's still lazy
        assert hasattr(ds_nws_lazy.aerosol_type.data, 'dask')

        # Check decodings (this triggers compute)
        assert ds_nws_lazy.aerosol_type.values[0] == 'Dust'
        assert ds_nws_lazy.constituent_type.values[0] == 'Ozone'
        assert ds_nws_lazy.source_sink_indicator.values[0] == 'Sink'


def test_open_mfdataset_preprocess():
    """
    Test that open_mfdataset correctly applies the preprocess function.
    """
    with patch('grib2io.open') as mock_grib_open, patch(
        'grib2io.xarray_backend.msgs_from_index'
    ) as mock_msgs, patch(
        'grib2io.xarray_backend._open_dataset_from_index'
    ) as mock_open_from_idx:
        # Setup mock grib file object
        mock_f = MagicMock()
        mock_f._index = {'ny': [2], 'nx': [2]}
        mock_grib_open.return_value.__enter__.return_value = mock_f

        mock_msgs.return_value = [MagicMock()]

        # Mock the final dataset
        ds_mock = xr.Dataset({'TMP': (('y', 'x'), np.ones((2, 2)))})
        mock_open_from_idx.return_value = ds_mock

        # Preprocess function
        def preprocess(ds):
            ds.attrs['preprocessed'] = True
            return ds

        # Call open_mfdataset
        ds = xarray_backend.open_mfdataset(['file1.grib2'], preprocess=preprocess)

        # Check if preprocess was called
        assert ds.attrs['preprocessed'] is True

        # Verify that _open_dataset_from_index was called with a list of filenames
        args, kwargs = mock_open_from_idx.call_args
        assert isinstance(args[1], list)
        assert args[1] == ['file1.grib2']


def test_ptype_vectorization_and_laziness(mock_grib2io_backend):
    """
    Verify that PTYPE decoding works for both Eager (NumPy) and Lazy (Dask) data.
    """
    from grib2io.xarray_backend import _decode_ptype

    mock_lookup, _ = mock_grib2io_backend

    # Setup mock to return variable length strings
    mock_lookup.side_effect = (
        lambda val, tbl: 'Short' if str(val) == '1' else 'VeryLongString'
    )

    # 1. Eager check (NumPy)
    eager_data = np.array([1, 2])
    decoded_eager = _decode_ptype(eager_data)
    assert decoded_eager[1] == 'VeryLongString'
    assert decoded_eager[0] == 'Short'

    # 2. Lazy check (Dask)
    import dask.array as da

    lazy_data = da.from_array(eager_data, chunks=2)
    da_ptype = xr.DataArray(lazy_data, dims='x', name='threshold_lower_limit')

    # Apply _decode_ptype via apply_ufunc
    decoded_lazy = xr.apply_ufunc(
        _decode_ptype,
        da_ptype,
        dask='parallelized',
        output_dtypes=[np.dtypes.StringDType] if _HAS_STRINGDTYPE else [object],
    )

    assert decoded_lazy.chunks is not None
    # Verify result identity
    assert (decoded_lazy.compute().values == decoded_eager).all()


def test_scientific_provenance_initialization():
    """
    Verify that scientific provenance (history) is initialized during data load.
    """
    from grib2io.xarray_backend import GribBackendEntrypoint

    engine = GribBackendEntrypoint()

    # Mocking internal calls to avoid I/O
    with patch('grib2io.open'), patch('grib2io.xarray_backend.msgs_from_index'), patch(
        'grib2io.xarray_backend.parse_grib_index'
    ) as mock_parse, patch('grib2io.xarray_backend.make_variables') as mock_make, patch(
        'grib2io.xarray_backend.build_da_without_coords'
    ) as mock_build, patch('grib2io.xarray_backend.assign_xr_meta') as mock_assign:
        mock_parse.return_value = (MagicMock(), {}, {}, {})
        mock_make.return_value = (
            [pd.DataFrame({'shortName': ['TMP']})],
            [{'x': range(1), 'y': range(1)}],
            {},
        )

        mock_da = xr.DataArray([1.0], name='TMP')
        mock_build.return_value = mock_da

        mock_ds = xr.Dataset({'TMP': mock_da})
        mock_assign.return_value = mock_ds

        ds = engine.open_dataset('dummy.grib2')

        assert 'history' in ds.attrs
        assert 'Initialized via grib2io.open_dataset' in ds.attrs['history']
        assert 'UTC' in ds.attrs['history']
