from setuptools import setup, Extension
from pathlib import Path


EXT_PATH = Path("./src/extensions")
LIB_PATH = Path("./bin/shared")
extension_kwargs = {
    'library_dirs': [str(LIB_PATH)]
}

BH_DIR = EXT_PATH / "BHPairProduction"
bh_ext = Extension(
    "agnprocesses.ext.bh",
    sources=[str(BH_DIR / "bh.cpp")],
    libraries=["BetheHeitler"],
    **extension_kwargs,
)

GGIR_DIR = EXT_PATH / "GammaGammaInteractionRate"
ggir_ext = Extension(
    "agnprocesses.ext.ggir",
    sources=[str(GGIR_DIR / "gamma_gamma_interaction_rate_ext.cpp")],
    libraries=["GammaGammaInteractionRate"],
    **extension_kwargs,
)

GGPP_DIR = EXT_PATH / "GammaGammaPairProduction"
ggpp_ext = Extension(
    "agnprocesses.ext.ggpp",
    sources=[str(GGPP_DIR / "gamma-gamma_ext.cpp")],
    libraries=["GammaGammaPairProduction"],
    **extension_kwargs,
)

PGAMMA_DIR = EXT_PATH / "PhotoHadron"
pgamma_ext = Extension(
    "agnprocesses.ext.pgamma",
    sources=[str(PGAMMA_DIR / "pgamma_ext.cpp")],
    libraries=["PhotoHadron"],
    **extension_kwargs,
)

ICIR_DIR = EXT_PATH / "InverseComptonInteractionRate"
icir_ext = Extension(
    "agnprocesses.ext.icir",
    sources=[str(ICIR_DIR / "icir_ext.cpp")],
    libraries=["InverseComptonInteractionRate"],
    **extension_kwargs,
)


setup(
    name="agnprocesses",
    version="0.3",
    description="A toolbox for modelling processes in Active Galactic Nuclei",
    author="Egor Podlesniy, Timur Dzhatdoev, Igor Vaiman",
    author_email="podlesnyi.ei14@physics.msu.ru",
    license="GPLv3",
    package_dir={"": "src"},
    include_package_data=True,
    packages=["agnprocesses"],
    ext_modules=[bh_ext, ggir_ext, ggpp_ext, pgamma_ext, icir_ext],
    install_requires=['astropy', 'numpy', 'scipy'],
)
