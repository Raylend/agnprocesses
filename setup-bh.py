from distutils.core import setup, Extension
import subprocess

cmd = "echo $LD_LIBRARY_PATH"
cmdout = subprocess.check_output(cmd, shell=True)[:-1].decode('utf-8')

module = Extension(
    "bh_ext",
    sources=['processes/c_codes/BHPairProduction/bh.cpp'],
    # library_dirs=[cmdout],
    library_dirs=['/home/raylend/Science/agnprocesses/bin/shared'],
    libraries=['BetheHeitler']
)

setup(
    name='bh_ext',
    version='0.1.0',
    description="Implementation with an argument transfer of Dzhatdoev's Bethe-Heitler photohadron pair production library",
    ext_modules=[module]
)
