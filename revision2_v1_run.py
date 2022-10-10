from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
import agnprocesses.spectra as spec
import torch
import electromagnetic_cascades
import agnprocesses.ic as ic
import agnprocesses.gamma_gamma as gamma_gamma

device = 'cuda'
original_params = [1.0, 1.0, 1.0]
# folder = "Wendel_case2_test_v007_100_000_ic_losses_below_threshold=False"
# field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/Wendel_photon_field_case2.txt"
# field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/Wendel_photon_field_case1a.txt"
n_init_particles = 30_000
folder = "test80_NEW_MC_primary_gamma_field_30000_eps_thr=1e+05eV_no_losses_below_threshold_1e+09---1e+15eV_photon_max=None_further_than_60_per_cent_RBLR"
field_path = "data/PKS1510-089/nph"
# test78 --- after fixing [1, 0] to [1, :] in the escape filter

print(f"Ouput folder: {folder}")

print("Number of primary particles is ", n_init_particles)

ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
r_blr = (0.036 * u.pc).to(u.cm)  # BLR radius
print("r_blr = {:.3e}".format(r_blr))

l = 0.4 * r_blr  # maximum possible distance travelled by hadronic gamma rays
print("l = {:.3e}".format(l))

primary_energy_min = 1.0e+09 * u.eV
print("primary_energy_min = {:.6e}".format(primary_energy_min))
primary_energy_max = 1.0e+15 * u.eV
print("primary_energy_max = {:.6e}".format(primary_energy_max))

observable_energy_min = 7.0e+07 * u.eV
print("observable_energy_min = {:.3e}".format(observable_energy_min))
observable_energy_max = 1.0e+15 * u.eV
print("observable_energy_max = {:.3e}".format(observable_energy_max))

eps_thr = 1.0e+05 * u.eV
photon_max = None

print("energy_ic_threshold = {:.3e} eV".format(eps_thr.to(u.eV).value))
print("background energy of photon max = {}".format(
    photon_max)
)

electromagnetic_cascades.monte_carlo_process(
    n_init_particles,
    l,  # : astropy.Quantity # e.g., 1.0e+12 * u.cm
    field_path,
    injection_place='uniform',
    primary_spectrum=spec.power_law,
    primary_spectrum_params=original_params,
    primary_energy_min=primary_energy_min,
    primary_energy_max=primary_energy_max,
    energy_ic_threshold=eps_thr,
    primary_energy_tensor_user=None,
    primary_particle='gamma',  # or 'electron'
    observable_energy_min=observable_energy_min,
    observable_energy_max=observable_energy_max,
    background_photon_energy_unit=u.eV,
    background_photon_density_unit=(u.eV * u.cm**3)**(-1),
    energy_photon_max=photon_max,
    terminator_step_number=3000,
    folder=folder,
    ic_losses_below_threshold=False,
    device=device)
