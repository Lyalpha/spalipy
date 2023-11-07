from time import time
from typing import Any, Union

import mmap
import numpy as np
import tempfile


def _c_array_prep(array: Union[np.ndarray, np.memmap]) -> Union[np.ndarray, np.memmap]:
    """
    Return a numpy array for use in C code by ensuring it is contiguous and in native byte order
    """
    # When calling this function the return value should be assigned to the original array
    # e.g. array = _to_native_byte_order(array), since the array is modified in place
    if not array.data.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.byteorder in ("=", "|"):
        return array
    return array.newbyteorder("=").byteswap(inplace=False)


def _memmap_tryfree(obj: Any) -> bool:
    """
    Attempt to release memory usage from a np.memmap object. Return True on success, else False.
    """
    if isinstance(obj, np.memmap):
        try:
            obj._mmap.madvise(mmap.MADV_DONTNEED)
            return True
        except Exception:  # noqa
            pass
    return False


def _memmap_create_temp(ndarray_to_save: np.ndarray) -> np.memmap:
    """Create and return temporary file np.memmap object using defaults as per tempfile.
    Temp file will be unlinked on exit but will persist for use until memmap is garbage collected.
    """
    if not isinstance(ndarray_to_save, np.ndarray):
        raise ValueError("ndarray_to_save must be np.ndarray object.")
    tmp_file = tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=rf"spalipy_{int(time())}_",
        suffix=r".dat",
        delete=True,
    )
    tmp_file.write(ndarray_to_save.tobytes(order="C"))
    tmp_file.seek(0)
    tmp_memmap = np.memmap(
        tmp_file,
        mode="r+",
        dtype=ndarray_to_save.dtype,
        shape=ndarray_to_save.shape,
        order="C",
    )
    return tmp_memmap
