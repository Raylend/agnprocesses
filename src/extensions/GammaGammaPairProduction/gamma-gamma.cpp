#define PY_SSIZE_T_CLEAN
#include "/home/raylend/anaconda3/include/python3.7m/Python.h"

#include "gamma-gamma-core.h"

static PyObject *
pair(PyObject *self, PyObject *args)
{
    char *photon_file;
    char *gamma_file;
    //
    if (!PyArg_ParseTuple(args, "ss", &photon_file, &gamma_file))
        return NULL;
    //
    pair_calculate(photon_file, gamma_file);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"pair", (PyCFunction)pair, METH_VARARGS, "execute pair.c with photon field at photon_file and gamma-ray energy and SED at gamma_file. Computes energy and SED of electron-positron pairs."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef pair_ext = {
    PyModuleDef_HEAD_INIT,
    "pair_ext",
    "Implementation of Egor Podlesnyi's gamma-gamma pair production C codes. Input gamma-ray SED is considred as the SED of absorbed gamma-rays, so before using this code you should multiply your gamma-ray SED by (1 - exp(-tau)).",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_pair_ext(void)
{
    return PyModule_Create(&pair_ext);
}
