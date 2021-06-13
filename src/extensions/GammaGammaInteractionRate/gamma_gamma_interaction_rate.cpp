#define PY_SSIZE_T_CLEAN
#include "/home/raylend/anaconda3/include/python3.7m/Python.h"

#include "gamma_gamma_interaction_rate_core.h"

static PyObject *
rate(PyObject *self, PyObject *args)
{
    char *photon_file;
    double E_min;
    double E_max;
    //
    if (!PyArg_ParseTuple(args, "sdd", &photon_file, &E_min, &E_max))
        return NULL;
    //
    gamma_gamma_interaction_rate(photon_file, E_min, E_max);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"rate", (PyCFunction)rate, METH_VARARGS, "executes gamma_gamma_interaction_rate.c with photon field at photon_file and E_min, E_max as gamma-ray minimum and maximum energies. Computes gamma-gamma absorption interaction rate in 1/cm."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef gamma_gamma_interaction_rate_ext = {
    PyModuleDef_HEAD_INIT,
    "gamma_gamma_interaction_rate_ext",
    "Implementation of Egor Podlesnyi's gamma-gamma interaction rate computation C codes. Input photon field file is considered as energy in eV and spectral number density in 1 / (eV * cm^3). E_min and E_max are in eV.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_gamma_gamma_interaction_rate_ext(void)
{
    return PyModule_Create(&gamma_gamma_interaction_rate_ext);
}
