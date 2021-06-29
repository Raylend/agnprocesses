from __future__ import annotations

import numpy as np
from dataclasses import dataclass

import astropy.units as u
from astropy.units import Quantity
from astropy import constants as const

from typing import ClassVar, Optional, Union

from .utils import MaybeQuantity, validate_maybe_quantity, E_ratio_as_float


# for semantic typing
Energy = Quantity
Frequency = Quantity
Temperature = Quantity


class EnergyDependentQuantity:
    """
    Base class for all energy-dependent quantities, i.e. table functions of energy.
    Subclasses should specify default_unit class attribute so that it can be checked on intialization.
    """

    # to be redefined by subclasses
    default_unit: Optional[ClassVar[u.Unit]] = None

    def __init__(self, E: MaybeQuantity, q: MaybeQuantity) -> EnergyDependentQuantity:
        if self.__class__.default_unit is not None:
            self.q = self.__class__._validate_with_default_unit(q)
        elif isinstance(q, Quantity):
            self.q = q
        else:
            raise TypeError(
                'q must be a astropy.units.Quantity when initializing a class '
                + f'without default_unit attribute, but {q.__class__.__name__} was passed'
            )
        self.E = validate_maybe_quantity(E, u.eV)

        # boring validations of size, shape etc
        if self.E.ndim != 1 or self.q.ndim != 1:
            raise ValueError(
                f"Only 1-D Energy Dependent Quantities are supported, but {self.E.ndim = }, {self.q.ndim = }"
            )
        if self.E.size != self.q.size:
            raise ValueError(
                f"Energy and quantity must be astropy.Quantity of the same size, but {self.E.size = }, {self.q.size = }!"
            )
        if np.unique(self.E).size < self.E.size:
            raise ValueError(f"Energy array cannot contain duplicates")

    @property
    def unit(self) -> u.Unit:
        return self.q.unit

    @property
    def size(self) -> int:
        return self.E.size

    @classmethod
    def _validate_with_default_unit(cls, v: MaybeQuantity) -> Quantity:
        if cls.default_unit is None:
            if isinstance(v, Quantity):
                return v
            else:
                raise TypeError(
                    "Cannot infer units in class with no default_unit attribute, "
                    + "specify all units manually or use more specific subclass"
                )
        else:
            return validate_maybe_quantity(v, cls.default_unit)

    @classmethod
    def from_txt(cls, filename: str, E_unit: Optional[u.Unit], q_unit: Optional[u.Unit]) -> EnergyDependentQuantity:
        data = np.loadtxt(filename)
        E = data[:, 0] * E_unit
        q = data[:, 1] * q_unit
        return cls(E, q)

    @classmethod
    def from_numpy(
        cls, table: 'Nx2 NumpyArray', E_unit: Optional[u.Unit] = None, q_unit: Optional[u.Unit] = None  # type: ignore
    ) -> EnergyDependentQuantity:
        E = table[:, 0]
        if E_unit is not None:
            E *= E_unit
        q = table[:, 1]
        if q_unit is not None:
            q *= q_unit
        return cls(E, q)

    def to_numpy(
        self, E_unit: Optional[u.Unit] = None, q_unit: Optional[u.Unit] = None
    ) -> 'Nx2 NumpyArray':  # type: ignore
        E = self.E.to(E_unit or self.E.unit).value
        q = self.q.to(q_unit or self.q.unit).value
        return np.concatenate((E[..., np.newaxis], q[..., np.newaxis]), axis=1)

    # constructors for standard laws
    # TODO: make separate class for analytical functions

    @classmethod
    def power_law(
        cls, E: Energy, gamma: float, norm: MaybeQuantity = 1.0, E_ref: Energy = 1.0 * u.eV
    ) -> EnergyDependentQuantity:
        return cls(E, cls._validate_with_default_unit(norm) * (E / E_ref) ** (-gamma))

    @classmethod
    def broken_power_law(
        cls,
        E: Energy,
        gamma_1: float,
        gamma_2: float,
        E_break: Energy,
        norm: Quantity,
    ):
        q = np.ones_like(E) * cls._validate_with_default_unit(norm)
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
        x = E_ratio_as_float(E, E_ref)
        return cls(E, cls._validate_with_default_unit(norm) * x ** (-(alpha + beta * np.log10(x))))

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
            cls._validate_with_default_unit(norm)
            * E_ratio_as_float(E, E_ref) ** (-gamma)
            * np.exp(-E_ratio_as_float(E, E_cutoff)),
        )


class SEDPerAreaTime(EnergyDependentQuantity):
    """Spectral energy distribution (E^2 dN / dE dS dt)"""

    default_unit = u.eV * u.cm**(-2) * u.s**(-1)


class SEDPerTime(EnergyDependentQuantity):
    """Spectral energy distribution (E^2 dN / dE dt)"""

    default_unit = u.eV * u.s**(-1)


class SpatialSpectralPhotonDensity(EnergyDependentQuantity):
    """Quantity of photons per volume per energy unit"""

    default_unit = 1.0 / (u.eV * u.cm**3)

    @classmethod
    def greybody(
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
