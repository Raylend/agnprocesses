"""
This program computes (electron + positron) SED
produced via gamma-ray - photon collisions using
E. I. Podlesnyi's C codes.
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


def gamma_gamma_install(dev_mode=True):
    try:
        with open('processes/gamma_gamma-log', mode='r') as f:
            gamma_gamma_install_flag = int(f.read(1))
            f.close()
    except:
        gamma_gamma_install_flag = 0
    if gamma_gamma_install_flag == 0 or dev_mode == True:
        # %% 1. creating .o files
        print("1. Creating .o files...")
        ########################################################################
        cmd = "g++ -c -fPIC processes/c_codes/GammaGammaPairProduction/gamma-gamma-core.cpp -o bin/shared/gamma-gamma-core.o"
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        ########################################################################
        cmd = "g++ -c -fPIC processes/c_codes/GammaGammaPairProduction/gamma-gamma.cpp -o bin/shared/gamma-gamma.o"
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        ########################################################################
        print('Done!')
        # % % 2. creating a library file .so
        print("2. Creating an .so library file...")
        cmd = 'g++ -shared bin/shared/gamma-gamma-core.o bin/shared/gamma-gamma.o -o bin/shared/libGammaGammaPairProduction.so'
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        print('Done!')
        # %% 3. installing setup.py, i.e. installing the module
        print("3. Installing the module...")
        cmd = 'python setup-gamma-gamma.py install'
        cmdout = subprocess.check_output(cmd, shell=True)[:-1]
        print(str(cmdout))
        print('Done!')
        # %% 4. Completed
        print("4.Installation of gamma-gamma pair production library completed.")
        with open('processes/gamma_gamma-log', mode='w') as f:
            f.write('1')
            f.close()
    else:
        pass
    return None


def pair_production(field,
                    gamma,
                    background_photon_energy_unit=u.eV,
                    background_photon_density_unit=(u.eV * u.cm**3)**(-1),
                    gamma_energy_unit=u.eV,
                    gamma_sed_unit=u.eV / (u.cm**2 * u.s)):
    """
    field is the string with the path to the target photon field .txt file
    OR numpy array with 2 colums: the first column is the background photon
    energy, the second columnn is the background photon density.
    Units in the field table must correspond to the
    background_photon_energy_unit parameter and the
    background_photon_density_unit parameter.
    field should contain no more than 1000 strings (rows)!!!
    (more strings will be implemented in future)

    gamma is the string with the path to the gamma-ray energy/SED .txt file
    OR numpy array with 2 colums: the first column is the gamma-ray
    energy, the second columnn is the gamma-ray SED.
    NB: gamma should contain SED of gamma-rays to be absorbed, i.e. the should
    be multiplied by (1 - exp(-tau))!
    Units in the gamma table must correspond to the gamma_energy_unit parameter
    and the gamma_sed_unit parameter.
    gamma mustn't contain more than 5000 strings (rows)!
    This restriction will be removed in future updates.

    Returns a tuple with electron + positron energy in
    gamma_energy_unit and SED in gamma_sed_unit
    as array-like astropy Quantities.
    """
    try:
        energy_coef = background_photon_energy_unit.to(u.eV) / (1.0 * u.eV)
        dens_coef = background_photon_density_unit.to(
            (u.eV * u.cm**3)**(-1)) / (u.eV * u.cm**3)**(-1)
    except AttributeError:
        raise AttributeError(
            "Make sure that background_photon_energy_unit is in energy units, background_photon_density_unit is in [energy * volume]**(-1) units.")
    ###########################################################################
    try:
        energy_gamma_coef = gamma_energy_unit.to(u.eV) / (1.0 * u.eV)
    except AttributeError:
        raise AttributeError(
            "Make sure that gamma_energy_unit is in energy units.")
    ###########################################################################
    gamma_gamma_install()
    import pair_ext
    ###########################################################################
    # background photon field
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
    if field[:, 0].shape[0] > 1000:
        raise NotImplementedError(
            "field should contain no more than 1000 strings (rows)! (more strings will be implemented in future)")
    photon_path = 'processes/c_codes/GammaGammaPairProduction/input/photon_path.txt'
    np.savetxt(photon_path, field, fmt='%.6e')
    ###########################################################################
    # gamma-ray SED
    if type(gamma) == type(''):
        try:
            gamma = np.loadtxt(gamma)
            gamma[:, 0] = gamma[:, 0] * energy_gamma_coef
            gamma[:, 1] = gamma[:, 1]
        except:
            raise ValueError(
                "Cannot read 'gamma'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / SED).\nTry to use an absolute path.")
    elif type(gamma) == type(np.array(([2, 1], [5, 6]))):
        gamma[:, 0] = gamma[:, 0] * energy_gamma_coef
        gamma[:, 1] = gamma[:, 1]
    else:
        raise ValueError(
            "Invalid value of 'gamma'! Make sure it is a numpy array \n with 2 columns or a string with the path to a .txt file with \n 2 columns (energy / sed).")
    if gamma[:, 0].shape[0] > 5000:
        raise NotImplementedError(
            "gamma should contain no more than 5000 strings (rows)! (more strings will be implemented in future)")
    gamma_path = 'processes/c_codes/GammaGammaPairProduction/input/gamma_path.txt'
    np.savetxt(gamma_path, gamma, fmt='%.6e')
    ###########################################################################
    pair_ext.pair(photon_path, gamma_path)
    pair = np.loadtxt(
        'processes/c_codes/GammaGammaPairProduction/output/SED_gamma-gamma_pairs.txt')
    pair_e = (pair[:, 0] * u.eV).to(gamma_energy_unit)
    pair_sed = pair[:, 1] * gamma_sed_unit
    return (pair_e, pair_sed)


def test():
    print("gamma_gamma.py imported successfully.")
    return None
