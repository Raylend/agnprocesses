from distutils.core import setup, Extension

module = Extension(
    "pgamma_ext",
    sources=['processes/c_codes/PhotoHadron/pgamma.cpp'],
    library_dirs=[
        '/home/raylend/Science/agnprocesses/bin/shared'],
    libraries=['PhotoHadron']
)

setup(
    name='pgamma_ext',
    version='0.0.1',
    description="Initial trial to implement Dzhatdoev's PhotoHadron library",
    ext_modules=[module]
)
