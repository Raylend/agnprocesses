import pytest
from pytest import param

import numpy as np
import astropy.units as u

from agnprocesses.data_classes.energy_dependent_quantities import EnergyDependentQuantity, SED
from agnprocesses.data_classes.utils import validate_maybe_quantity, E_ratio_as_float


@pytest.mark.parametrize(
    'E_in, q_in, E, q',
    [
        param(
            np.array([1, 2, 3]) * u.eV,
            np.array([4, 5, 6]) * u.m,
            np.array([1, 2, 3]) * u.eV,
            np.array([4, 5, 6]) * u.m,
            id='fully qualified init',
        ),
        param(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]) * u.Hz,
            np.array([1, 2, 3]) * u.eV,
            np.array([4, 5, 6]) * u.Hz,
            id='numpy array of energies is interpreted as eV'
        ),
        param(
            np.array([1, 2, 3]) * u.J,
            np.array([4, 5, 6]) * u.K * u.m * u.s**(-1),
            np.array([1, 2, 3]) * u.J,
            np.array([4, 5, 6]) * u.K * u.m * u.s**(-1),
            id='non-standard energy units are preserved'
        ),
    ]
)
def test_energy_dependent_quantity_init(E_in, q_in, E, q):
    eq = EnergyDependentQuantity(E_in, q_in)
    for expected, inclass in zip([E, q], [eq.E, eq.q]):
        assert expected.unit == inclass.unit
        assert np.all(np.isclose(expected, inclass))


@pytest.mark.parametrize(
    'E_in, q_in, errortype',
    [
        param(
            np.array([1, 2, 3]) * u.eV,
            np.array([4, 5, 6]),
            TypeError,
            id='no units specified for quantity'
        ),
        param(
            np.array([1, 2, 3]) * u.Hz,
            np.array([4, 5, 6]) * u.m,
            ValueError,
            id='incovertible energy units'
        ),
    ]
)
def test_failed_energy_dependent_quantity_init(E_in, q_in, errortype):
    with pytest.raises(errortype):
        eq = EnergyDependentQuantity(E_in, q_in)


@pytest.mark.parametrize(
    'E_in, q_in, E, q',
    [
        param(
            np.array([1, 2, 3]) * u.eV,
            np.array([4, 5, 6]),
            np.array([1, 2, 3]) * u.eV,
            np.array([4, 5, 6]) * u.eV * u.cm**(-2) * u.s**(-1),
            id='SED numpy array is converted to default units'
        ),
        param(
            np.array([1, 2, 3]) * u.TeV,
            np.array([4, 5, 6]) * u.TeV * u.m**(-2) * u.hour**(-1),
            np.array([1, 2, 3]) * u.TeV,
            np.array([4, 5, 6]) * u.TeV * u.m**(-2) * u.hour**(-1),
            id='non-standard SED units are preserved'
        ),
    ]
)
def test_SED_init(E_in, q_in, E, q):
    sed = SED(E_in, q_in)
    for expected, inclass in zip([E, q], [sed.E, sed.q]):
        assert expected.unit == inclass.unit
        assert np.all(expected == inclass)


@pytest.mark.parametrize(
    'mq, unit, output',
    [
        param(1, u.eV, 1 * u.eV),
        param(1 * u.J, u.eV, 1 * u.J),
        param(1 * u.m, u.eV, None),
        param(np.array([1, 2, 3]) * u.m, u.km, np.array([1, 2, 3]) * u.m),
        param(np.array([1, 2, 3]), u.Hz, np.array([1, 2, 3]) * u.Hz),
        param(np.array([1, 2, 3]) * u.s, u.m, None),
    ]
)
def test_maybe_quantity_validation(mq, unit, output):
    if output is not None:
        assert np.all(validate_maybe_quantity(mq, unit) == output)
    else:
        with pytest.raises(ValueError):
            validate_maybe_quantity(mq, unit)


@pytest.mark.parametrize(
    'E_num, E_den, output',
    [
        param(1 * u.eV, 1 * u.eV, 1),
        param(1 * u.GeV, 1 * u.eV, 10**9),
        param(1 * u.keV, 5 * u.PeV, 0.2 * 10**(-12)),
    ]
)
def test_E_ratio_as_float(E_num, E_den, output):
    assert E_ratio_as_float(E_num, E_den) == output
