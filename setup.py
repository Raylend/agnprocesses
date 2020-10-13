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
    version='0.2.0',
    description="Implementation with argument transfer of Dzhatdoev's PhotoHadron library",
    ext_modules=[module]
)
