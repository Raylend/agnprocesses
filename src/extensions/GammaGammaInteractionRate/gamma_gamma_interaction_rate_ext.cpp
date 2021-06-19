#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "gamma_gamma_interaction_rate.h"

static PyObject *
rate(PyObject *self, PyObject *args)
{
    char *photon_file;
    char *output_file;
    double E_min;
    double E_max;

    if (!PyArg_ParseTuple(args, "ssdd", &photon_file, &output_file, &E_min, &E_max))
        return NULL;

    gamma_gamma_interaction_rate(photon_file, output_file, E_min, E_max);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"rate", (PyCFunction)rate, METH_VARARGS, "executes gamma_gamma_interaction_rate.c with photon field at photon_file and E_min, E_max as gamma-ray minimum and maximum energies. Computes gamma-gamma absorption interaction rate in 1/cm."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef ggir_module = {
    PyModuleDef_HEAD_INIT,
    "ggir",
    "Implementation of Egor Podlesnyi's gamma-gamma interaction rate computation C codes. Input photon field file is considered as energy in eV and spectral number density in 1 / (eV * cm^3). E_min and E_max are in eV.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_ggir(void)
{
    return PyModule_Create(&ggir_module);
}
