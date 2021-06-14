BINDIR= ./bin

${BINDIR}:
	mkdir -p ${BINDIR}

LIBDIR = ../../../bin/shared

${LIBDIR}:
	mkdir -p ${LIBDIR}


OBJECT_FILE = ${BINDIR}/PhotoHadron.o

${OBJECT_FILE}: ./PhotoHadron.cpp | ${BINDIR}
	g++ -c -fPIC $< -o $@


LIB_FILE = ${LIBDIR}/libPhotoHadron.so

${LIB_FILE}: ${OBJECT_FILE} | ${LIBDIR}
	g++ -shared $^ -o $@


PHONY: lib

lib: ${LIB_FILE}
