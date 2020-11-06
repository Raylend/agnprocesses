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
    "bh_ext",
    sources=['processes/c_codes/BHPairProduction/bh.cpp'],
    library_dirs=[agn_path],
    # library_dirs=['/home/raylend/Science/agnprocesses/bin/shared'],
    libraries=['BetheHeitler']
)

setup(
    name='bh_ext',
    version='0.1.0',
    description="Implementation with an argument transfer of Dzhatdoev's Bethe-Heitler photohadron pair production library",
    ext_modules=[module]
)
