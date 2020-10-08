#define PY_SSIZE_T_CLEAN
#include "/home/raylend/anaconda3/include/python3.7m/Python.h"

#include "PhotoHadron.h"

static PyObject *
pgamma(PyObject *self)
{
    // printf("%s\n", str1);
    photohadron();
    // FILE * fp;
    // fp = fopen("test_file.out", "w");
    // if (fp == NULL)
    // {
    //     printf("Cannot create test_file.out!\n");
    //     exit(1);
    // }
    // char test_ok[10] = "TEST: OK";
    // fputs(test_ok, fp);
    // fclose(fp);
    Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    {"pgamma", (PyCFunction)pgamma, METH_NOARGS, "execute PhotoHadron.cpp"},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static struct PyModuleDef pgamma_ext = {
    PyModuleDef_HEAD_INIT,
    "pgamma_ext",
    "Initial test implementation of Dzhatdoev's photohadron C++ codes",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_pgamma_ext(void)
{
    return PyModule_Create(&pgamma_ext);
}
