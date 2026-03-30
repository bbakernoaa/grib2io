import base64
import logging
from typing import Dict, List
import json

import fsspec
import numcodecs
import numpy as np
import zarr

import grib2io

logger = logging.getLogger("grib2io.kerchunk")

class GRIB2IOCodec(numcodecs.abc.Codec):
    """
    Read GRIB stream of bytes as a message using grib2io
    """
    codec_id = "grib2io"

    def __init__(self, var, dtype=None):
        self.var = var
        self.dtype = dtype

    def encode(self, buf):
        return buf

    def decode(self, buf, out=None):
        import io
        with grib2io.open(bytes(buf), mode='r') as f:
            msg = f[0]
            if self.var in ["latitude", "longitude"]:
                lats, lons = msg.latlons()
                data = lats if self.var == "latitude" else lons
                dt = self.dtype or "float64"
            else:
                data = msg.data
                dt = self.dtype or "float32"

            if out is not None:
                return numcodecs.compat.ndarray_copy(data, out)
            else:
                return data.astype(dt, copy=False)

numcodecs.register_codec(GRIB2IOCodec, "grib2io")

def _encode_for_JSON(obj):
    if hasattr(obj, "to_bytes") and not isinstance(obj, (int, float, str)):
        obj = obj.to_bytes()
    if isinstance(obj, dict):
        return {k: _encode_for_JSON(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_encode_for_JSON(v) for v in obj]
    elif isinstance(obj, bytes):
        return "base64:" + base64.b64encode(obj).decode("ascii")
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

def scan_grib(url, storage_options=None, inline_threshold=100, skip=0, filter=None) -> List[Dict]:
    """
    Generate references for a GRIB2 file using grib2io.
    """
    storage_options = storage_options or {}
    logger.debug(f"Open {url}")

    out = []
    with fsspec.open(url, "rb", **storage_options) as f:
        # We need the local file path if f is a local file, or a file-like object
        try:
            name = f.path if hasattr(f, 'path') else url
        except:
            name = url

        # Read file with grib2io
        # Note grib2io.open expects either a physical file or bytes, or file-like
        # We can pass `f` directly if it supports read/seek/tell, which fsspec does.
        # But `grib2io` needs an explicit file handle or file name to be efficient.
        # So we pass `f`
        with grib2io.open(f) as grib_file:
            for i, msg in enumerate(grib_file):
                if skip > 0 and i < skip:
                    continue

                # Filter
                good = True
                filter = filter or {}
                for k, v in filter.items():
                    if not hasattr(msg, k):
                        good = False
                        break
                    val = getattr(msg, k)
                    if isinstance(v, (list, tuple, set)):
                        if val not in v:
                            good = False
                            break
                    elif val != v:
                        good = False
                        break
                if not good:
                    continue

                # We have a valid message.
                # Find its offset and size
                offset = grib_file._index['offset'][i]
                size = grib_file._index['msgSize'][i]

                # Setup Zarr group for this message
                store_dict = {}
                z = zarr.open_group(store=store_dict, mode='w', zarr_format=2)

                # Add attributes
                attrs = {}
                for sect in [0, 1, 3, 4, 5]:
                    attrs.update(msg.attrs_by_section(sect, values=True))
                # stringify attributes that are numpy types or non-serializable
                for k, v in attrs.items():
                    if isinstance(v, np.ndarray):
                        attrs[k] = v.tolist()
                    elif isinstance(v, (np.generic, int, float, str)):
                        try:
                            attrs[k] = v.item()
                        except:
                            pass
                    else:
                        attrs[k] = str(v)

                z.attrs.update(attrs)

                varName = msg.shortName
                if not varName or varName in ("undef", "unknown"):
                    varName = "values"

                shape = (int(msg.ny), int(msg.nx))
                dtype = np.dtype("float32") if msg.typeOfValues == 0 else np.dtype("int32")

                # Data Variable Array
                d = z.create_array(
                    name=varName,
                    shape=shape,
                    chunks=shape,
                    dtype=dtype,
                    fill_value=getattr(msg, "priMissingValue", np.nan),
                    filters=[GRIB2IOCodec(var="values", dtype=str(dtype))],
                    compressor=None,
                )
                d.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
                store_dict[f"{varName}/0.0"] = ["{{u}}", int(offset), int(size)]

                # Coordinates
                for coord in ["latitude", "longitude"]:
                    c = z.create_array(
                        name=coord,
                        shape=shape,
                        chunks=shape,
                        dtype="float64",
                        fill_value=np.nan,
                        filters=[GRIB2IOCodec(var=coord, dtype="float64")],
                        compressor=None,
                    )
                    c.attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
                    store_dict[f"{coord}/0.0"] = ["{{u}}", int(offset), int(size)]

                z.attrs["coordinates"] = "latitude longitude"


                # Output dictionary
                # translating references to serializable
                out_dict = {
                    "version": 1,
                    "refs": _encode_for_JSON(store_dict),
                    "templates": {"u": url},
                }
                out.append(out_dict)

    return out



from kerchunk.combine import MultiZarrToZarr
from collections import defaultdict
import ujson

def grib_tree(message_groups: List[Dict], remote_options=None) -> Dict:
    """
    Build a hierarchical data model from a set of scanned grib messages using grib2io conventions.
    """
    zarr_store_dict = {}
    from kerchunk.utils import dict_to_store, translate_refs_serializable
    zarr_store = dict_to_store(zarr_store_dict)
    zroot = zarr.open_group(store=zarr_store, zarr_format=2)

    aggregations = defaultdict(list)

    for msg_ind, group in enumerate(message_groups):
        if not group.get("version") == 1:
            continue

        gattrs = group["refs"].get(".zattrs", "{}")
        try:
            gattrs = ujson.loads(gattrs) if isinstance(gattrs, (str, bytes)) else gattrs
        except:
            gattrs = {}

        # Find the data variable
        vname = None
        for key in group["refs"].keys():
            name = key.split("/")[0]
            if name not in [".zattrs", ".zgroup", "latitude", "longitude", "step", "time", "valid_time"] and not name.startswith("typeOf"):
                vname = name
                break

        if vname is None:
            continue

        dattrs = group["refs"].get(f"{vname}/.zattrs", "{}")
        try:
            dattrs = ujson.loads(dattrs) if isinstance(dattrs, (str, bytes)) else dattrs
        except:
            dattrs = {}

        zgroup = zroot.require_group(vname)
        if "name" not in zgroup.attrs:
            zgroup.attrs["name"] = dattrs.get("fullName", vname)

        # Set the coordinates attribute for the group
        zgroup.attrs["coordinates"] = gattrs.get("coordinates", "latitude longitude")
        # add to the list of groups to multi-zarr
        aggregations[zgroup.path].append(group)

    concat_dims = []
    identical_dims = ["y", "x"]
    for path in aggregations.keys():
        catdims = concat_dims.copy()
        idims = identical_dims.copy()

        mzz = MultiZarrToZarr(
            aggregations[path],
            remote_options=remote_options,
            concat_dims=catdims,
            identical_dims=idims,
        )
        group = mzz.translate()

        for key, value in group["refs"].items():
            if key not in [".zattrs", ".zgroup"]:
                zarr_store_dict[f"{path}/{key}"] = value

    zarr_dict = {
        key: (val.decode() if isinstance(val, bytes) else val)
        for key, val in zarr_store_dict.items()
    }
    translate_refs_serializable(zarr_dict)
    return dict(refs=zarr_dict, version=1)
