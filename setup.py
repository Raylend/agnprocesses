from setuptools import setup, Extension
from pathlib import Path

EXT_PATH = Path('src/extensions')
LIB_PATH = Path('bin/shared')

BH_DIR = EXT_PATH / 'BHPairProduction'
bh_ext = Extension(
    'agnprocesses.ext.bh',
    sources=[str(BH_DIR / 'bh.cpp')],
    library_dirs=[str(LIB_PATH)],
    libraries=['BetheHeitler'],
)

# module = Extension(
#     "bh_ext",
#     sources=['processes/c_codes/BHPairProduction/bh.cpp'],
#     library_dirs=[agn_path],
#     # library_dirs=['/home/raylend/Science/agnprocesses/bin/shared'],
#     libraries=['BetheHeitler']
# )

setup(
    name='agnprocesses',
    version='0.2',
    description='A toolbox for modelling processes in Active Galactic Nuclei',
    author='Egor Podlesniy',
    author_email='podlesnyi.ei14@physics.msu.ru',
    license='GPLv3',
    package_dir = {'': 'src'},
    packages=['agnprocesses'],
    ext_modules=[bh_ext]
)