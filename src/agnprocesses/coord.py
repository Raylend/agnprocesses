from astropy import units as u
from astropy import constants as const
import numpy as np
from astropy.coordinates import get_sun
from astropy.time import Time
import pandas as pd


# see https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone.pdf
MET_REFERENCE_DATE = pd.Timestamp('January 1, 2001, 0:0:0')


def fermi_met_to_utc(met):
    return (MET_REFERENCE_DATE + pd.Timedelta(met, unit='s')).to_numpy()


def get_sun_ra_fermi(t_fermi_sec):
    """
    Returns ra of the Sun at the moment of t_fermi_sec (MET).
    MET is the Mission Elapsed Time ('January 1, 2001, 0:0:0').

    t_fermi_sec must be numpy array or float.
    """
    try:
        s = t_fermi_sec.shape
        t = np.array(list(map(
            fermi_met_to_utc, t_fermi_sec)))
        t = Time(t)
        x = get_sun(t)
        return((x.frame.ra.to(u.deg)).value)
    except AttributeError:
        t = Time(fermi_met_to_utc(t_fermi_sec))
        x = get_sun(t)
        return((x.frame.ra.to(u.deg)).value)


def get_sun_dec_fermi(t_fermi_sec):
    """
    Returns dec of the Sun at the moment of t_fermi_sec (MET).
    MET is the Mission Elapsed Time ('January 1, 2001, 0:0:0').

    t_fermi_sec must be numpy array or float.
    """
    try:
        s = t_fermi_sec.shape
        t = np.array(list(map(
            fermi_met_to_utc, t_fermi_sec)))
        t = Time(t)
        x = get_sun(t)
        return((x.frame.dec.to(u.deg)).value)
    except AttributeError:
        t = Time(fermi_met_to_utc(t_fermi_sec))
        x = get_sun(t)
        return((x.frame.dec.to(u.deg)).value)


def ra_to_deg(hour, arcmin, arcsec):
    return((hour + (1.0 / 60.0 * arcmin) + (1.0 / 3600.0 * arcsec)) * 15.0)


def dec_to_deg(deg, arcmin, arcsec):
    return(deg + 1.0 / 60.0 * arcmin + 1.0 / 3600.0 * arcsec)
