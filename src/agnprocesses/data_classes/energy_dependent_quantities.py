from __future__ import annotations

import numpy as np
from dataclasses import dataclass

import astropy.units as u
from astropy.units import Quantity
from astropy import constants as const

from typing import ClassVar, Optional, Union
from numbers import Number
from nptyping import NDArray


# for semantic typing
Energy = Quantity
Frequency = Quantity
Temperature = Quantity
MaybeQuantity = Union[Quantity, Number, NDArray]


def E_ratio_to_float(E_num: MaybeQuantity, E_den: MaybeQuantity) -> float:
    x = E_num / E_den
    if isinstance(x, Quantity):
        x = x.to(u.dimensionless_unscaled).value
    return x


def validate_maybe_quantity(v: MaybeQuantity, unit: u.Unit) -> Quantity:
    if not isinstance(v, Quantity):
        return v * unit  # works both on numpy arrays and single floats/ints
    else:
        try:
            v.to(unit)
        except u.core.UnitConversionError:
            raise ValueError(
                f"Quantity is inconvertible to the unit ({unit})"
            )
        return v


class EnergyDependentQuantity:
    """
    Base class for all energy-dependent quantities. Subclasses should specify default_unit class
    attribute so that it can be checked on intialization.
    """

    # to be redefined by subclasses
    default_unit: Optional[ClassVar[u.Unit]] = None

    def __init__(self, E: MaybeQuantity, q: MaybeQuantity) -> EnergyDependentQuantity:
        if self.__class__.default_unit is not None:
            self.q = self.__class__._validate_maybe_quantity(q)
        elif isinstance(q, Quantity):
            self.q = q
        else:
            raise TypeError(
                'q must be a astropy.units.Quantity when initializing a class '
                + f'without default_unit attribute, but {q.__class__.__name__} was passed'
            )
        self.E = validate_maybe_quantity(E, u.eV)

        if E.ndim != 1 or q.ndim != 1:
            raise ValueError(
                f"Only 1-D Energy Dependent Quantities are supported, but {self.E.ndim = }, {self.q.ndim = }"
            )
        if E.size != q.size:
            raise ValueError(
                f"Energy and quantity must be astropy.Quantity of the same size, but {self.E.size = }, {self.q.size = }!"
            )

    @property
    def unit(self) -> u.Unit:
        return self.q.unit

    @classmethod
    def _validate_maybe_quantity(cls, v: MaybeQuantity) -> Quantity:
        if cls.default_unit is None:
            raise TypeError(
                "Cannot infer units in class with no default_unit attribute, "
                + "specify all units manually or use more specific subclass"
            )
        return validate_maybe_quantity(v, cls.default_unit)

    # standard law constructors
    # TODO: make separate class for analytical functions

    @classmethod
    def power_law(
        cls, E: Energy, gamma: float, norm: MaybeQuantity = 1.0, E_ref: Energy = 1.0 * u.eV
    ) -> EnergyDependentQuantity:
        return cls(E, cls._validate_maybe_quantity(norm) * (E / E_ref) ** (-gamma))

    @classmethod
    def broken_power_law(
        cls,
        E: Energy,
        gamma_1: float,
        gamma_2: float,
        E_break: Energy,
        norm: Quantity,
    ):
        q = np.ones_like(E) * cls._validate_maybe_quantity(norm)
        q[E <= E_break] *= (E[E <= E_break] / E_break) ** (-gamma_1)
        q[E > E_break] *= (E[E <= E_break] / E_break) ** (-gamma_2)
        return cls(E, q)

    @classmethod
    def log_parabola(
        cls,
        E: Energy,
        alpha: float,
        beta: float,
        norm: MaybeQuantity = 1.0,
        E_ref: Energy = 1.0 * u.eV,
    ):
        x = E_ratio_to_float(E, E_ref)
        return cls(E, cls._validate_maybe_quantity(norm) * x ** (-(alpha + beta * np.log10(x))))

    @classmethod
    def exponential_cutoff(
        cls,
        E: Energy,
        gamma: float,
        E_cutoff: Energy,
        norm: MaybeQuantity = 1.0,
        E_ref: Energy = 1.0 * u.eV,
    ):
        return cls(
            E,
            cls._validate_maybe_quantity(norm)
            * E_ratio_to_float(E, E_ref) ** (-gamma)
            * np.exp(-E_ratio_to_float(E, E_cutoff)),
        )


class SED(EnergyDependentQuantity):
    default_unit = u.eV * u.cm**(-2) * u.s**(-1)

    @property
    def sed(self) -> Quantity:
        return self.q


# TODO: name for this class!
class PhotonsPerEnergyPerVolume(EnergyDependentQuantity):
    default_unit = 1.0 / (u.eV * u.cm**3)

    @classmethod
    def greybody_spectrum(
        cls, E: Union[Energy, Frequency], temperature: Union[Energy, Temperature], dilution: float = 1.0
    ):
        KT = temperature.to(u.J, equivalencies=u.temperature_energy())
        nu_characteristic = KT / const.h
        nu = E.to(u.Hz, equivalencies=u.spectral())
        x = (nu / nu_characteristic).to(u.dimensionless_unscaled)
        s = (8 * np.pi * const.h * nu**3) / const.c**3 / (np.exp(x) - 1)
        s /= const.h
        s /= E
        s *= dilution
        return cls(E=E, q=s.to(cls.default_unit))
