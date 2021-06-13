BINDIR= ./bin

${BINDIR}:
	mkdir -p ${BINDIR}

LIBDIR = ../../../bin/shared

${LIBDIR}:
	mkdir -p ${LIBDIR}


OBJECT_FILE = ${BINDIR}/GGIR.o

${OBJECT_FILE}: ./gamma_gamma_interaction_rate.c | ${BINDIR}
	g++ -c -fPIC $< -o $@


LIB_FILE = ${LIBDIR}/libGammaGammaInteractionRate.so

${LIB_FILE}: ${OBJECT_FILE} | ${LIBDIR}
	g++ -shared $^ -o $@


PHONY: lib

lib: ${LIB_FILE}
