"""
This program computes (electron + positron) Bethe-Heitler SED
produced via proton-photon collisions using T. A. Dzhatdoev's C++ codes.
It includes only electron + positron pair production,
it does not include photomeson production and its secondaries.
For the latter process see pgamma.py file in this package.
See Kelner, & Aharonian Phys. Rev. D 78, 034013 (2008).
"""
import subprocess  # to run prompt scripts from python
# import os
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
# import matplotlib.pyplot as plt
try:
    import processes.spectra as spec
except:
    try:
        import spectra as spec
    except:
        raise ImportError("a problem with importing spectra.py occured")


def bh_install(dev_mode=True):
    try:
        with open('processes/logs/bh-log', mode='r') as f:
            bh_install_flag = int(f.read(1))
            f.close()
    except:
        bh_install_flag = 0
    if bh_install_flag == 0 or dev_mode == True:
        # %% 1. creating .o files
        print("1. Creating .o files...")
        ########################################################################
        cmd = "g++ -c -fPIC processes/c_codes/BHPairProduction/Pair.cpp -o bin/shared/Pair.o"
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        ########################################################################
        cmd = "g++ -c -fPIC processes/c_codes/BHPairProduction/bh.cpp -o bin/shared/bh.o"
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        ########################################################################
        print('Done!')
        # % % 2. creating a library file .so
        print("2. Creating an .so library file...")
        cmd = 'g++ -shared bin/shared/Pair.o bin/shared/bh.o -o bin/shared/libBetheHeitler.so'
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        print('Done!')
        # %% 3. installing setup.py, i.e. installing the module
        print("3. Installing the module...")
        cmd = 'python setup-bh.py install'
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        print(str(cmdout))
        print('Done!')
        # %% 4. Completed
        print("4.Installation of Bethe-Heitler pair production library completed.")
        with open('processes/logs/bh-log', mode='w') as f:
            f.write('1')
            f.close()
    else:
        pass
    return None


def kelner_bh_calculate(field,
                        energy_proton_min,
                        energy_proton_max,
                        p_p,
                        e_cut_p=-1,
                        C_p=1.0 / (u.eV),
                        background_photon_energy_unit=u.eV,
                        background_photon_density_unit=(u.eV * u.cm**3)**(-1)):
    """
    energy_proton_min is the minimum proton energy
    (must be an astropy Quantity of energy or float (in the latter case it will
    be considered as Lorentz factor))

    energy_proton_max is the maximum proton energy
    (must be the same type as energy_proton_min)

    e_cut_p is the cutoff proton energy. It must be an astropy Quantity or
    float (in the latter case it will be considered as Lorentz factor).
    If it is less than 0 (default), an ordinary power law spectrum will be used.

    field is the string with the path to the target photon field .txt file
    OR numpy array with 2 colums: the first column is the background photon
    energy, the second columnn is the background photon density.
    Units in the field table must correspond to the
    background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.

    field should contain no more than 100 strings (rows)!!!
    (more strings will be implemented in future)

    C_p is the normalization coefficient of the proton spectrum.

    Returns a tuple with electron + positron energy in
    eV and SED in eV * s**(-1) as array-like astropy Quantities.
    """
    try:
        energy_coef = background_photon_energy_unit.to(u.eV) / (1.0 * u.eV)
        dens_coef = background_photon_density_unit.to(
            (u.eV * u.cm**3)**(-1)) / (u.eV * u.cm**3)**(-1)
    except AttributeError:
        raise AttributeError(
            "Make sure that background_photon_energy_unit is in energy units, background_photon_density_unit is in [energy * volume]**(-1) units.")
    bh_install()
    import bh_ext
    ###########################################################################
    if type(field) == type(''):
        try:
            field = np.loadtxt(field)
            field[:, 0] = field[:, 0] * energy_coef
            field[:, 1] = field[:, 1] * dens_coef
        except:
            raise ValueError(
                "Cannot read 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).\nTry to use an absolute path.")
    elif type(field) == type(np.array(([2, 1], [5, 6]))):
        field[:, 0] = field[:, 0] * energy_coef
        field[:, 1] = field[:, 1] * dens_coef
    else:
        raise ValueError(
            "Invalid value of 'field'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / density).")
    if field[:, 0].shape[0] > 100:
        raise NotImplementedError(
            "field should contain no more than 100 strings (rows)! (more strings will be implemented in future)")
    proton_target_path = 'processes/c_codes/BHPairProduction/input/field.txt'
    np.savetxt(proton_target_path, field, fmt='%.6e')
    ###########################################################################
    try:
        energy_proton_min = energy_proton_min.to(u.eV)
        energy_proton_max = energy_proton_max.to(u.eV)
    except AttributeError:
        try:
            energy_proton_min = (energy_proton_min * const.m_p * const.c**2).to(
                u.eV)
            energy_proton_max = (energy_proton_max * const.m_p * const.c**2).to(
                u.eV)
        except:
            raise ValueError(
                "Problems with energy_proton_min and energy_proton_max!\nMake sure they are astropy Quantities or float!")
    try:
        e_cut_value = e_cut_p.value
    except AttributeError:
        if e_cut_p < 0:
            e_cut_p = e_cut_p * u.dimensionless_unscaled
        else:
            e_cut_p = (e_cut_p * const.m_p * const.c**2).to(u.eV)
    ###########################################################################
    bh_ext.bh(proton_target_path,
              energy_proton_min.value,
              energy_proton_max.value,
              p_p, e_cut_p.value)
    ###########################################################################
    pair = np.loadtxt(
        'processes/c_codes/BHPairProduction/output/BH_SED.txt')
    pair_e = pair[:, 0] * u.eV
    pair_sed = pair[:, 1] * (u.eV * u.s**(-1))
    pair_sed = pair_sed * C_p / (1.0 / u.eV)
    return (pair_e, pair_sed)


def test():
    print("bh.py imported successfully.")
    return None
