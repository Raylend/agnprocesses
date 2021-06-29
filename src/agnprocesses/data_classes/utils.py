import astropy.units as u
from astropy.units import Quantity

from typing import Union
from numbers import Number


# number or numpy array, convertible to quantity
MaybeQuantity = Union[Quantity, Number, 'NumpyArray']  # type: ignore


def validate_maybe_quantity(v: MaybeQuantity, unit: u.Unit) -> Quantity:
    if not isinstance(v, Quantity):
        return v * unit  # works both on numpy arrays and single floats/ints!
    else:
        v.to(unit)  # checking convertability
        return v


def E_ratio_as_float(E_num: MaybeQuantity, E_den: MaybeQuantity) -> float:
    ratio = validate_maybe_quantity(E_num, u.eV) / validate_maybe_quantity(E_den, u.eV)
    return ratio.to(u.dimensionless_unscaled).value
