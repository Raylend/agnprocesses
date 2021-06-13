BINDIR= ./bin

${BINDIR}:
	mkdir -p ${BINDIR}

LIBDIR = ../../../bin/shared

${LIBDIR}:
	mkdir -p ${LIBDIR}


PAIR_OBJECT_FILE = ${BINDIR}/BetheHeitler.o

${PAIR_OBJECT_FILE}: ./BetheHeitler.cpp | ${BINDIR}
	g++ -c -fPIC $< -o $@


LIB_FILE = ${LIBDIR}/libBetheHeitler.so

${LIB_FILE}: ${PAIR_OBJECT_FILE} ./src/P01Pair.cpp ./src/P01Structures.cpp | ${LIBDIR}
	g++ -shared $^ -o $@


PHONY: lib

lib: ${LIB_FILE}
