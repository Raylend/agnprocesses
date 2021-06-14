BINDIR= ./bin

${BINDIR}:
	mkdir -p ${BINDIR}

LIBDIR = ../../../bin/shared

${LIBDIR}:
	mkdir -p ${LIBDIR}


OBJECT_FILE = ${BINDIR}/PhotoHadron.o

${OBJECT_FILE}: ./PhotoHadron.cpp | ${BINDIR}
	g++ -c -fPIC $< -I./src -o $@


LIB_FILE = ${LIBDIR}/libPhotoHadron.so

${LIB_FILE}: ${OBJECT_FILE} | ${LIBDIR}
	g++ -shared $^ -o $@


PHONY: lib test test_reset

lib: ${LIB_FILE}

# temporary - testing refactoring

test_reset:
	rm -rf ${BINDIR}

test: ./PhotoHadron.cpp | test_reset ${BINDIR}
	g++ $^ -o ${BINDIR}/PH && ${BINDIR}/PH

