from distutils.core import setup, Extension
import subprocess

cmd = "echo $LD_LIBRARY_PATH"
cmdout = subprocess.check_output(cmd, shell=True)[:-1].decode('utf-8')
agn_path = None
try:
    cmdout = cmdout.split(':')
    for path in cmdout:
        if 'agnprocesses/bin/shared' in path:
            agn_path = path
            break
except:
    raise ImportError(
        "$LD_LIBRARY_PATH is not defined properly. See README for instructions")

if agn_path is None:
    raise ImportError(
        "$LD_LIBRARY_PATH is not defined properly. See README for instructions")

module = Extension(
    "gamma_gamma_interaction_rate_ext",
    sources=[
        'processes/c_codes/GammaGammaInteractionRate/gamma_gamma_interaction_rate.cpp'],
    library_dirs=[agn_path],
    # library_dirs=['/home/raylend/Science/agnprocesses/bin/shared'],
    libraries=['GammaGammaInteractionRate']
)

setup(
    name='gamma_gamma_interaction_rate_ext',
    version='0.1.0',
    description="Implementation of Egor Podlesnyi's gamma-gamma interaction rate computation C codes. Input photon field file is considered as energy in eV and spectral number density in 1 / (eV * cm^3). E_min and E_max are in eV.",
    ext_modules=[module]
)
