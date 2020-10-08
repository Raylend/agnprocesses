# This is the main python3 file for agnprocesses package.
"""
agnprocessec is the package for calculation of various astrophysical
processes which can occur in active galactic nuclei.
It includes:
- synchrotron emission
- inverse Compton (IC) process
- photo-hadron process
- and more
"""
# %% import
import processes.synchro as synchro
import processes.ic as ic
import processes.spectra as spec
from astropy import units as u
from astropy import constants as const
import numpy as np
import matplotlib.pyplot as plt
import subprocess  # to run prompt scripts from python


# def p_gamma_prepare():
#     cmd = 'pwd'
#     cmdout = (subprocess.check_output(
#         cmd,
#         shell=True)[:-1]).decode("utf-8") + '/bin/shared'
#     cmd = "export LD_LIBRARY_PATH=%s" % cmdout
#     print("cmd = {}".format(cmd))
#     cmdout = (subprocess.check_output(cmd, shell=True)[:-1]).decode("utf-8")
#     cmd = "echo $LD_LIBRARY_PATH"
#     cmdout = (subprocess.check_output(cmd, shell=True)[:-1]).decode("utf-8")
#     print("$LD_LIBRARY_PATH modified:")
#     print(cmdout)
#     return None


def pgamma_install():
    # p_gamma_prepare()
    # %% 1. creating .o files
    print("1. Creating .o files...")
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01Structures.cpp -o bin/shared/B01Structures.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    ###########################################################################
    cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/PhotoHadron.cpp -o bin/shared/PhotoHadron.o"
    cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    ###########################################################################
    cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/pgamma.cpp -o bin/shared/pgamma.o"
    cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    ###########################################################################
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronG.cpp -o bin/shared/B01PhotoHadronG.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronP.cpp -o bin/shared/B01PhotoHadronP.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronE.cpp -o bin/shared/B01PhotoHadronE.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronNuMu.cpp -o bin/shared/B01PhotoHadronNuMu.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronAntiNuMu.cpp -o bin/shared/B01PhotoHadronAntiNuMu.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronNuE.cpp -o bin/shared/B01PhotoHadronNuE.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PhotoHadronAntiNuE.cpp -o bin/shared/B01PhotoHadronAntiNuE.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01PlanckianCMB.cpp -o bin/shared/B01PlanckianCMB.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01Planckian.cpp -o bin/shared/B01Planckian.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    # cmd = "g++ -c -fPIC processes/c_codes/PhotoHadron/src/B01SSC.cpp -o bin/shared/B01SSC.o"
    # cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    print('Done!')
    # % % 2. creating a library file .so
    print("2. Creating an .so library file...")
    cmd = 'g++ -shared bin/shared/PhotoHadron.o bin/shared/pgamma.o -o bin/shared/libPhotoHadron.so'
    # cmd = 'gcc -shared bin/shared/pgamma.o -o bin/shared/libPhotoHadron.so'
    cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    print('Done!')
    # %% 3. installing setup.py, i.e. installing the module
    print("3. Installing the module...")
    cmd = 'python setup.py install'
    cmdout = subprocess.check_output(cmd, shell=True)[:-1]
    print(str(cmdout))
    print('Done!')
    # # # %% 4. cheking
    # # print("4.Checking")
    return None


if __name__ == '__main__':
    pgamma_install()
    import pgamma_ext
    pgamma_ext.pgamma()
    # %% pre-check
    # cmd = 'which python'
    # cmdout = subprocess.check_output(cmd, shell=True)
    # print(cmdout)
    # spec.test()
    # synchro.test()
    # ic.test()
    # # %% run
    # energy_proton_min = 1.0e+17 * u.eV
    # energy_proton_max = 1.0e+22 * u.eV
    # p_p = 2.0
    # en_cut = 2.960774e+19 * u.eV
    # pgamma.write_proton_parameters(energy_proton_min,
    #                                energy_proton_max,
    #                                p_p,
    #                                en_cut=en_cut)
    # cmdout = pgamma.run_cpp_photo_hadron()
    # print(cmdout)
    # %%
    # b = (1.0 * u.g**0.5 * u.cm**(-0.5) * u.s**(-1))
    # # b = 3 * u.G.decompose().cgs
    # print("b = {}".format(b))
    # norm = 1.0 * u.GeV**(-1)
    # particle_mass = const.m_e.cgs
    # particle_charge = const.e.gauss
    # omega_B_0 = particle_charge * b / (particle_mass * const.c.cgs)
    #
    # en_cutoff = 1.0 * u.GeV  # / particle_mass.to(u.GeV, u.mass_energy())
    # omega_0 = omega_B_0 * 4.0 / 3.0 * en_cutoff**2
    # nu_0 = omega_0 / (2.0 * np.pi)
    # nu_B_0 = omega_B_0 / (2.0 * np.pi)
    # nu = np.logspace(10, 15, 10) * u.Hz
    # f = synchro.derishev_synchro_spec(nu, b=b, norm=norm,
    #                                   spec_law='exponential_cutoff',
    #                                   gamma1=3.0,
    #                                   en_cutoff=en_cutoff,
    #                                   en_min=1.0e-04 * en_cutoff,
    #                                   en_max=1.0e+02 * en_cutoff,
    #                                   en_ref=en_cutoff)
    #
    # print(f)
    # SED = (const.h**2 * nu**2).to(u.eV**2) * f
    # SED = SED / np.max(SED)
    # nu = nu / en_cutoff**2 / nu_B_0
    # Derishev = np.loadtxt('Derishev_fig4.out')
    ############################################################################
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(1, 1, 1)
    #
    # plt.plot(
    #     nu, SED,
    #     marker=None,
    #     linewidth=3,
    #     # color = 'g',
    #     label='Egor'
    # )
    # plt.plot(
    #     Derishev[:, 0], Derishev[:, 1],
    #     marker=None,
    #     linewidth=3,
    #     linestyle='--',
    #     # color = 'b',
    #     label='Derishev'
    # )
    #
    # plt.xscale("log")
    # plt.yscale("log")
    # #ax.set_xlim(1.0e+04, 1.0e+05)
    # #ax.set_ylim(1.0e-09, 1.0e-07)
    # # ax.grid()
    # # ax.grid()
    # plt.legend(loc='upper right')
    # fig.savefig(
    #     'test_figures/exponential_cutoff_compare_with_Derishev_fig4a.pdf'
    # )
    #
    # plt.show()
