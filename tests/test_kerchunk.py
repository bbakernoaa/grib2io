import pytest
import xarray as xr
import numpy as np
import fsspec

try:
    import importlib.util

    importlib.util.find_spec("kerchunk")
    from grib2io.kerchunk import scan_grib, grib_tree

    HAS_KERCHUNK = True
except ImportError:
    HAS_KERCHUNK = False

pytestmark = pytest.mark.skipif(not HAS_KERCHUNK, reason="kerchunk is not available")


def test_scan_grib(request):
    data = request.config.rootdir / "tests" / "input_data" / "blend.t00z.core.f001.tmp.co.grib2"

    # Generate kerchunk references
    refs = scan_grib(str(data))

    # We expect 1 message in this specific blend file.
    assert len(refs) == 1

    ref = refs[0]
    assert ref["version"] == 1
    assert "refs" in ref
    assert "templates" in ref

    # Validate some key attributes exist
    store = ref["refs"]
    assert "TMP/.zarray" in store
    assert "TMP/.zattrs" in store

    # Validate the coordinate mappings exist
    assert "latitude/.zarray" in store
    assert "longitude/.zarray" in store

    # Validate that data blocks exist
    assert "TMP/0.0" in store

    # Test reading the file using fsspec+zarr
    mapper = fsspec.get_mapper("reference://", fo=ref)
    ds = xr.open_zarr(mapper, consolidated=False)

    assert "TMP" in ds.data_vars
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    assert ds.TMP.shape == (1597, 2345)

    # Read some data to trigger the codec
    val = ds.TMP.values[100, 100]
    assert not np.isnan(val)


def test_grib_tree(request):
    data = request.config.rootdir / "tests" / "input_data" / "blend.t00z.core.f001.tmp.co.grib2"

    refs = scan_grib(str(data))

    # Combine references into a tree
    tree = grib_tree(refs)

    assert tree["version"] == 1
    assert "refs" in tree

    # Verify that the references have been rewritten to the root level or grouped
    store = tree["refs"]
    assert "TMP/TMP/.zarray" in store
    assert "TMP/TMP/0.0" in store
