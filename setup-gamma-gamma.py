from distutils.core import setup, Extension
import subprocess

cmd = "echo $LD_LIBRARY_PATH"
cmdout = subprocess.check_output(cmd, shell=True)[:-1].decode('utf-8')

module = Extension(
    "pair_ext",
    sources=['processes/c_codes/GammaGammaPairProduction/gamma-gamma.cpp'],
    # library_dirs=[cmdout],
    library_dirs=['/home/raylend/Science/agnprocesses/bin/shared'],
    libraries=['GammaGammaPairProduction']
)

setup(
    name='pair_ext',
    version='0.1.0',
    description="Implementation of Egor Podlesnyi's gamma-gamma pair production C codes. Input gamma-ray SED is considred as the SED of absorbed gamma-rays, so before using this code you should multiply your gamma-ray SED by (1 - exp(-tau)).",
    ext_modules=[module]
)
