from dask import array as da
import numpy as np
import pytest

from iris.util import has_mask


def make_masked(
    with_mask=True,
    dask=False,
):
    data = [1, 2]
    mask = [False, False]
    if with_mask:
        if dask:
            func = da.ma.masked_array
        else:
            func = np.ma.masked_array
        result = func(data, mask=mask)
    else:
        if dask:
            func = da.array
        else:
            func = np.array
        result = func(data)

    return result


@pytest.mark.parametrize(
    "with_mask, expected",
    ([False, False], [True, True]),
    ids=["without_mask", "with_mask"],
)
@pytest.mark.parametrize(
    "dask", [False, True], ids=["numpy_array", "dask_array"]
)
def test_all(with_mask, dask, expected):
    array = make_masked(with_mask, dask)
    assert has_mask(array) == expected
