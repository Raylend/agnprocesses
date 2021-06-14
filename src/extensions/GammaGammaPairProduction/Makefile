BINDIR= ./bin

${BINDIR}:
	mkdir -p ${BINDIR}

LIBDIR = ../../../bin/shared

${LIBDIR}:
	mkdir -p ${LIBDIR}


OBJECT_FILE = ${BINDIR}/GGPP.o

${OBJECT_FILE}: ./pairs.c | ${BINDIR}
	g++ -c -fPIC $< -o $@


LIB_FILE = ${LIBDIR}/libGammaGammaPairProduction.so

${LIB_FILE}: ${OBJECT_FILE} | ${LIBDIR}
	g++ -shared $^ -o $@


PHONY: lib

lib: ${LIB_FILE}
