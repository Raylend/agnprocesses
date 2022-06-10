#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "IC_interaction_rate.h"

static PyObject *
icir(PyObject *self, PyObject *args)
{
    char *photon_file;
    double E_min;
    double E_max;
    double E_thr;
    //
    if (!PyArg_ParseTuple(args, "sddd", &photon_file, &E_min, &E_max, &E_thr))
        return NULL;
    //
    inverse_compton_interaction_rate(photon_file, E_min, E_max, E_thr);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"icir", (PyCFunction)icir, METH_VARARGS, "Executes IC_interaction_rate.c with photon field at photon_file and E_min, E_max as gamma-ray minimum and maximum energies. Computes Inverse Compton (IC) interaction rate in 1/cm given the minimum energy threshold of a secondary electron as E_thr."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef icir_module = {
    PyModuleDef_HEAD_INIT,
    "icir",
    "Implementation of Egor Podlesnyi's IC interaction rate computation C codes. Input photon field file is considered as energy in eV and spectral number density in 1 / (eV * cm^3). E_thr, E_min and E_max are in eV.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_icir(void)
{
    return PyModule_Create(&icir_module);
}
