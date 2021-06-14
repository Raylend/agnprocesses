#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "pairs.h"

static PyObject *
pair(PyObject *self, PyObject *args)
{
    char *photon_file;
    char *gamma_file;
    char *output_file;
    //
    if (!PyArg_ParseTuple(args, "sss", &photon_file, &gamma_file, &output_file))
        return NULL;
    //
    pair_calculate(photon_file, gamma_file, output_file);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"pair", (PyCFunction)pair, METH_VARARGS, "execute pair.c with photon field at photon_file and gamma-ray energy and SED at gamma_file. Computes energy and SED of electron-positron pairs."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef pair_ext_module = {
    PyModuleDef_HEAD_INIT,
    "pair",
    "Implementation of Egor Podlesnyi's gamma-gamma pair production C codes. Input gamma-ray SED is considred as the SED of absorbed gamma-rays, so before using this code you should multiply your gamma-ray SED by (1 - exp(-tau)).",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_pair_ext(void)
{
    return PyModule_Create(&pair_ext_module);
}
