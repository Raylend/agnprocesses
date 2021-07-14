#define PY_SSIZE_T_CLEAN
#include "/home/raylend/anaconda3/include/python3.7m/Python.h"

#include "IC_interaction_rate_core.h"

static PyObject *
IC_rate(PyObject *self, PyObject *args)
{
    char *photon_file;
    double E_min;
    double E_max;
    double E_thr;
    //
    if (!PyArg_ParseTuple(args, "sddd", &photon_file, &E_min, &E_max, &E_thr))
        return NULL;
    //
    IC_interaction_rate(photon_file, E_min, E_max, E_thr);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"IC_rate", (PyCFunction)IC_rate, METH_VARARGS, "Executes IC_interaction_rate.c with photon field at photon_file and E_min, E_max as gamma-ray minimum and maximum energies. Computes Inverse Compton (IC) interaction rate in 1/cm given the minimum energy threshold of a secondary electron as E_thr."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef IC_interaction_rate_ext = {
    PyModuleDef_HEAD_INIT,
    "IC_interaction_rate_ext",
    "Implementation of Egor Podlesnyi's IC interaction rate computation C codes. Input photon field file is considered as energy in eV and spectral number density in 1 / (eV * cm^3). E_thr, E_min and E_max are in eV.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_IC_interaction_rate_ext(void)
{
    return PyModule_Create(&IC_interaction_rate_ext);
}
