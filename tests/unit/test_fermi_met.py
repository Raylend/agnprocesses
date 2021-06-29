import pytest

import numpy as np
from agnprocesses.coord import fermi_met_to_utc


@pytest.mark.parametrize(
    'met, expected_utc',
    [
        pytest.param(0, np.datetime64('2001-01-01T00:00:00')),
        pytest.param(633538888, np.datetime64('2021-01-28T15:01:28')),
    ]
)
def test_fermi_met_to_utc(met, expected_utc):
    assert fermi_met_to_utc(met) == expected_utc
