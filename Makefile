PHONY: bh_lib install clean reinstall

bh_lib:
	cd src/extensions/BHPairProduction && make lib

ggir_lib:
	cd src/extensions/GammaGammaInteractionRate && make lib


install: bh_lib ggir_lib
	python setup.py install

clean:
	pip uninstall -y agnprocesses && rm -rf bin build dist src/agnprocesses.egg-info

reinstall: clean | install
