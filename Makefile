PHONY: bh_lib ggir_lib icir_lib ggpp_lib ph_lib install clean reinstall

bh_lib:
	cd src/extensions/BHPairProduction && make lib

ggir_lib:
	cd src/extensions/GammaGammaInteractionRate && make lib

ggpp_lib:
	cd src/extensions/GammaGammaPairProduction && make lib

ph_lib:
	cd src/extensions/PhotoHadron && make lib

icir_lib:
	cd src/extensions/InverseComptonInteractionRate && make lib

install: bh_lib ggir_lib ggpp_lib ph_lib icir_lib
	python setup.py install
	mkdir -p data/torch

clean:
	pip uninstall -y agnprocesses && rm -rf bin build dist src/agnprocesses.egg-info

reinstall: clean | install
