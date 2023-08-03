import numpy as np


def _c_array_prep(array):
    """Return a numpy array for use in C code by ensuring it is contiguous and in native byte order"""
    # When calling this function the return value should be assigned to the original array
    # e.g. array = _to_native_byte_order(array), since the array is modified in place
    if not array.data.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.byteorder in ("=", "|"):
        return array
    else:
        return array.newbyteorder("=").byteswap(inplace=False)
