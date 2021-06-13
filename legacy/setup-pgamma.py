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
    "pgamma_ext",
    sources=['processes/c_codes/PhotoHadron/pgamma.cpp'],
    library_dirs=[agn_path],
    # library_dirs=['/home/raylend/Science/agnprocesses/bin/shared'],
    libraries=['PhotoHadron']
)

setup(
    name='pgamma_ext',
    version='0.3.0',
    description="Implementation with argument transfer of Dzhatdoev's PhotoHadron library and .py wrapper-like function kelner_pgamma_calculate()",
    ext_modules=[module]
)
