#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "PhotoHadron.h"

static PyObject *
pgamma(PyObject *self, PyObject *args)
{
    char *file_path;
    char *output_path;
    double energy_proton_min, energy_proton_max;
    double p_p;
    double E_cut;
    //
    if (!PyArg_ParseTuple(args, "ssdddd", &file_path, &output_path, &energy_proton_min, &energy_proton_max, &p_p, &E_cut))
        return NULL;
    //
    photohadron(file_path, output_path, energy_proton_min, energy_proton_max, p_p, E_cut);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"pgamma", (PyCFunction)pgamma, METH_VARARGS, "execute PhotoHadron.cpp with photon field at file_path and given proton spectrum parameters"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef pgamma_ext = {
    PyModuleDef_HEAD_INIT,
    "pgamma_ext",
    "Implementation of Dzhatdoev's photohadron C++ codes",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_pgamma_ext(void)
{
    return PyModule_Create(&pgamma_ext);
}
