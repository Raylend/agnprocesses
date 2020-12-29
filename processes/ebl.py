"""
This program computes extragalactic gamma-photon absorption on the
extragalactic background light (EBL).
Currently available EBL models:
1) Gilmore et al. Mon. Not. R. Astron. Soc. 422, 3189–3207 (2012)
2) to be added
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
from scipy import interpolate


def tau_gilmore(energy, redshift):
    """
    Returns tau array (with shape corresponding to energy shape)
    according to Gilmore et al. (2012) EBL model.

    energy is astropy Quantity (array-like or not)

    redshift is float
    """
    try:
        energy = energy.to(u.eV)
    except:
        raise ValueError("energy must be in any astropy energy units!")
    tau = np.loadtxt("processes/EBL_models/Tau-G12")
    z_tau = tau[0, 1:]
    e_tau = (tau[1:, 0] * u.TeV).to(u.eV)
    TAU = tau[1:, 1:]
    z_filter = (redshift < z_tau)
    order = np.arange(0, z_filter.shape[0], 1)
    TAU = TAU[:, order[z_filter][0]].reshape(np.max(e_tau.shape))
    tau_list = list()
    try:
        for element_e in energy:
            e_filter = (element_e < e_tau)
            if not any(e_filter):
                tau_list.append(
                    TAU[order[-1]]
                )
                print("Warning!" +
                      "energy is more than maximum possible! Applying tau for 100 TeV")
                # raise ValueError(
                #     "energy = {:.3e} is outside possible limits!".format(
                #         element_e))
            else:
                tau_list.append(
                    TAU[order[e_filter][0]]
                )
    except TypeError:
        e_filter = (energy < e_tau)
        if not any(e_filter):
            tau_list.append(
                TAU[order[-1]]
            )
            print("Warning!" +
                  "energy is more than maximum possible! Applying tau for 100 TeV")
            # raise ValueError(
            #     "energy = {:.3e} is outside possible limits!".format(
            #         energy))
        else:
            tau_list.append(
                TAU[order[e_filter][0]]
            )
    except IndexError:
        e_filter = (energy < e_tau)
        if not any(e_filter):
            tau_list.append(
                TAU[order[-1]]
            )
            print("Warning!" +
                  "energy is more than maximum possible! Applying tau for 100 TeV")
            # raise ValueError(
            #     "energy = {:.3e} is outside possible limits!".format(
            #         energy))
        else:
            tau_list.append(
                TAU[order[e_filter][0]]
            )
    return np.array(tau_list)
