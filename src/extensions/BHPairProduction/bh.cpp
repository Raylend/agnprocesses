#define PY_SSIZE_T_CLEAN
#include "Python.h"

#include "BetheHeitler.hpp"

static PyObject *
bh(PyObject *self, PyObject *args)
{
    char* photon_field_file_path;
    char* output_file_path;
    double energy_proton_min, energy_proton_max;
    double p_p;
    double E_cut;
    //
    if (!PyArg_ParseTuple(
        args,
        "ssdddd",
        &photon_field_file_path, &output_file_path,
        &energy_proton_min, &energy_proton_max,
        &p_p, &E_cut
    ))
        return NULL;
    //
    bh_pair_production(photon_field_file_path, output_file_path, energy_proton_min, energy_proton_max, p_p, E_cut);
    //
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {
        "bh",
        (PyCFunction)bh,
        METH_VARARGS,
        "execute Pair::Process with photon field at file_path and given proton spectrum parameters. Computes Bethe-Heitler electron + positron pair production spectrum."
    },
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef bh_module = {
    PyModuleDef_HEAD_INIT,
    "bh",
    "Implementation of Dzhatdoev's Bethe-Heitler pair production C++ codes",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_bh(void)
{
    return PyModule_Create(&bh_module);
}
