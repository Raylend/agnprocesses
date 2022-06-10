BINDIR= ./bin

${BINDIR}:
	mkdir -p ${BINDIR}

LIBDIR = ../../../bin/shared

${LIBDIR}:
	mkdir -p ${LIBDIR}


OBJECT_FILE = ${BINDIR}/ICIR.o

${OBJECT_FILE}: ./IC_interaction_rate.c | ${BINDIR}
	g++ -c -fPIC $< -o $@


LIB_FILE = ${LIBDIR}/libInverseComptonInteractionRate.so

${LIB_FILE}: ${OBJECT_FILE} | ${LIBDIR}
	g++ -shared $^ -o $@


PHONY: lib

lib: ${LIB_FILE}
