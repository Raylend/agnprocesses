"""
Took 4 days, 7:11:50.918525
"""
from astropy import units as u
from astropy import constants as const
import numpy as np
from scipy.integrate import simps
import agnprocesses.spectra as spec
import torch
# import test_pytorch_run
import test_pytorch
import agnprocesses.ic as ic
import agnprocesses.gamma_gamma as gamma_gamma

device = 'cuda'
original_params = [1.0, 1.0, 1.0]
# folder = "Wendel_case2_test_v007_100_000_ic_losses_below_threshold=False"
# field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/Wendel_photon_field_case2.txt"
# field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/Wendel_photon_field_case1a.txt"
folder = "test56_primary_gamma_field_6000_eps_thr=1e+06eV_no_losses_below_threshold_1e+10---10e+14.77eV_photon_max=None"
field = "data/PKS1510-089/nph"

ELECTRON_REST_ENERGY = const.m_e.to(u.eV, u.mass_energy()).value
r_blr = (0.036 * u.pc).to(u.cm)  # BLR radius
print("r_blr = {:.3e}".format(r_blr))
l = 0.4 * r_blr  # maximum possible distance travelled by hadronic gamma rays
print("l = {:.3e}".format(l))

primary_energy_min = 1.0e+10 * u.eV
print("primary_energy_min = {:.6e}".format(primary_energy_min))
primary_energy_max = 5.0e+14 * u.eV
print("primary_energy_max = {:.6e}".format(primary_energy_max))

observable_energy_min = 7.0e+07 * u.eV
print("observable_energy_min = {:.6e}".format(observable_energy_min))
observable_energy_max = 10.0**14.77 * u.eV
print("observable_energy_max = {:.6e}".format(observable_energy_max))

eps_thr = 1.0e+06 * u.eV

test_pytorch.monte_carlo_process(
    6_000,
    l,  # : astropy.Quantity # e.g., 1.0e+12 * u.cm
    field,
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
    energy_photon_max=None,
    terminator_step_number=3000,
    folder=folder,
    ic_losses_below_threshold=False,
    device=device)
###########################################################################
# # 0 Precalculate gamma-gamma interaction rate
# # field = "/home/raylend/Science/agnprocesses/data/test_1eV.txt"
# field = "/home/raylend/Science/agnprocesses/data/PKS1510-089/nph"
# energy_electron_node, r_IC_node = ic.IC_interaction_rate(
#     field,
#     1.0e+08 * u.eV,
#     1.0e+16 * u.eV,
#     3.0e+06 * u.eV,
#     background_photon_energy_unit=u.eV,
#     background_photon_density_unit=(u.eV * u.cm**3)**(-1)
# )
# print(f'energy_electron_node = {energy_electron_node}')
# print(f'r_IC_node = {r_IC_node}')
# Timur = np.loadtxt(
#     "data/Timur_cascades/Rates_gamma_electron.txt"
# )
# ###########################################################################
# fig, ax = plt.subplots(figsize=(8, 6))
# # Timur
# plt.plot(
#     Timur[:, 0], Timur[:, 2],
#     marker=None,
#     linestyle='-',
#     linewidth=3,
#     color='r',
#     zorder=100,
#     label='Timur'
# )
# # Egor
# plt.plot(
#     energy_electron_node, r_IC_node,
#     marker=None,
#     linestyle='--',
#     linewidth=3,
#     color='k',
#     zorder=100,
#     label='Egor'
# )
#
# plt.xlabel('energy, ' + str(energy_electron_node.unit), fontsize=18)
# plt.xticks(fontsize=12)
# plt.ylabel('Electron IC interaction rate, ' + '1 / cm', fontsize=18)
# plt.yticks(fontsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.legend()  # loc='upper left')
# plt.show()
#
# en = np.logspace(7, 9.5) * u.Hz
# grey = spec.greybody_spectrum(en, 8.0e-03 * u.K, dilution=1.0)
# # grey = grey * en**1
# # grey = grey / np.max(grey)
# grey = grey * const.c * const.h / np.pi * (const.h * en).to(u.eV) / 4
# grey = grey.to(u.W / (u.m**2 * u.Hz))
#
# en2 = np.logspace(0, 2.5) * u.Hz * 1.0e+12
# grey2 = spec.greybody_spectrum(en2, 1000 * u.K, dilution=1)
# # grey2 = grey2 * en2**1
# # grey2 = grey2 / np.max(grey2)
# grey2 = grey2 * const.c * const.h / np.pi * (const.h * en2).to(u.eV) / 4
# grey2 = grey2.to(u.W / (u.m**2 * u.Hz))
#
# elsevier = np.loadtxt("data/greybody_spectrum_recheck/elsevier.txt")
# # elsevier[:, 1] = elsevier[:, 1] / np.max(elsevier[:, 1])
# elsevier[:, 0] = elsevier[:, 0] * 1.0e+12
# elsevier[:, 1] = elsevier[:, 1] / 1.0e+12
#
#
# wiki = np.loadtxt("data/greybody_spectrum_recheck/wiki.txt")
# # wiki[:, 1] = wiki[:, 1] / np.max(wiki[:, 1])
#
# fig, ax = plt.subplots()
# # Egor
# plt.plot(
#     en, grey,
#     marker=None,
#     linestyle='--',
#     linewidth=3,
#     color='k',
#     zorder=100,
#     label='Egor'
# )
# # Wikipedia https://commons.wikimedia.org/wiki/File:RWP-comparison.svg
# plt.plot(
#     wiki[:, 0], wiki[:, 1],
#     marker=None,
#     linestyle='-',
#     linewidth=3,
#     color='g',
#     zorder=100,
#     label='Wikipedia'
# )
# # Egor
# # plt.plot(
# #     en2, grey2,
# #     marker=None,
# #     linestyle='--',
# #     linewidth=3,
# #     color='r',
# #     zorder=100,
# #     label='Egor'
# # )
# # # Elsevier https://www.sciencedirect.com/topics/engineering/blackbody-radiation
# # plt.plot(
# #     elsevier[:, 0], elsevier[:, 1],
# #     marker=None,
# #     linestyle='-',
# #     linewidth=3,
# #     color='b',
# #     zorder=100,
# #     label='Elsevier'
# # )
#
# plt.xlabel('frequency, ' + str(en.unit), fontsize=18)
# plt.xticks(fontsize=12)
# plt.ylabel('SED, ' + 'arb. units', fontsize=18)
# plt.yticks(fontsize=12)
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.legend(loc='upper left')
# plt.show()
#
#
# # z = 0.186
# # gamma = 1.5
# # E_cutoff = 10.0 * u.TeV
# #
# # E = np.logspace(-1, 7, 100) * u.MeV
# # SED_gamma = E**2 * spec.exponential_cutoff(E, gamma, E_cutoff)
# # tau_gamma_gamma = ebl.tau_gilmore(E, z)
# # d_l = cosmology.luminosity_distance(z)
# # print(f'luminosity_distance = {d_l.to(u.Mpc)}')
# # cosmology.test()
# #
# # SED_gamma_survived = SED_gamma * np.exp(-tau_gamma_gamma)
# # filter = (SED_gamma_survived >= 1.0e-04 * np.max(SED_gamma_survived))
# #
# # E_survived = E / (1.0 + z)
# # E_survived = E_survived[filter]
# # SED_gamma_survived = SED_gamma_survived[filter]
# #
# # n_photons_primary = simps(E.value, SED_gamma.value / E.value**2)
# # n_photons_survived = simps(E_survived.value,
# #                            SED_gamma.value / E_survived.value**2)
# #
# # SED_gamma_survived = SED_gamma_survived / \
# #     n_photons_survived * n_photons_primary
# #
# # fig, ax = plt.subplots()
# # # EBL absorption: OFF
# # plt.plot(
# #     E, SED_gamma,
# #     marker=None,
# #     linestyle='--',
# #     linewidth=3,
# #     # color='c',
# #     zorder=100,
# #     label='SED primary'
# # )
# # # EBL absorption: ON
# # plt.plot(
# #     E_survived, SED_gamma_survived,
# #     marker=None,
# #     linestyle='-',
# #     linewidth=3,
# #     # color='c',
# #     zorder=100,
# #     label='SED survived'
# # )
# #
# # plt.xlabel('energy, ' + str(E.unit), fontsize=18)
# # plt.xticks(fontsize=12)
# # plt.ylabel('SED, ' + str(SED_gamma_survived.unit), fontsize=18)
# # plt.yticks(fontsize=12)
# # ax.set_xscale('log')
# # ax.set_yscale('log')
# # plt.legend(loc='upper left')
# # plt.show()
