/* Optional CPU certificate for connected, three-dimensionally winding graphs.
 * Negative results are inconclusive: Python applies the full physical tests.
 * Buffers may be unaligned views into a typed-array record. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int64_t integer_at(const char *p, Py_ssize_t i) {
    int64_t value;
    memcpy(&value, p + 8 * i, 8);
    return value;
}

static double real_at(const char *p, Py_ssize_t i) {
    double value;
    memcpy(&value, p + 8 * i, 8);
    return value;
}

static Py_ssize_t root(Py_ssize_t *parent, Py_ssize_t i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
}

static PyObject *bulk(PyObject *self, PyObject *args) {
    PyObject *obj[4];
    Py_buffer buf[4] = {{0}};
    Py_ssize_t *parent = NULL;
    double cutoff2;
    int certified = 0;
    if (!PyArg_ParseTuple(args, "OOOOd", &obj[0], &obj[1], &obj[2], &obj[3], &cutoff2))
        return NULL;
    for (int k = 0; k < 4; ++k) {
        if (PyObject_GetBuffer(obj[k], &buf[k], PyBUF_FORMAT | PyBUF_STRIDES) < 0)
            goto done;
        const char *fmt = buf[k].format;
        if (*fmt == '=' || *fmt == '@') ++fmt;
        if (!PyBuffer_IsContiguous(&buf[k], 'C') || buf[k].ndim != 2 ||
            buf[k].itemsize != 8 ||
            (k < 2 ? strcmp(fmt, "q") && strcmp(fmt, "l") : strcmp(fmt, "d")))
            goto done;
    }
    Py_ssize_t n = buf[2].shape[0], e = buf[0].shape[1];
    if (n < 1 || e < n + 2 || buf[0].shape[0] != 2 ||
        buf[1].shape[0] != e || buf[1].shape[1] != 3 || buf[2].shape[1] != 3 ||
        buf[3].shape[0] != 3 || buf[3].shape[1] != 3)
        goto done;
    parent = PyMem_Malloc(n * sizeof(*parent));
    if (!parent) { PyErr_NoMemory(); goto done; }
    for (Py_ssize_t i = 0; i < n; ++i) parent[i] = i;
    Py_ssize_t components = n;
    int axes = 0;
    const char *edges = buf[0].buf, *shifts = buf[1].buf, *x = buf[2].buf;
    double cell[9];
    memcpy(cell, buf[3].buf, sizeof(cell));
    for (Py_ssize_t k = 0; k < e; ++k) {
        int axis = -1, multi = 0;
        int64_t h = 0;
        for (int d = 0; d < 3; ++d) {
            int64_t s = integer_at(shifts, 3*k+d);
            if (s) { if (axis >= 0) { multi = 1; break; } axis = d; h = s; }
        }
        if (multi || (axis >= 0 ? axes & (1 << axis) : components == 1)) continue;
        int64_t i = integer_at(edges, k), j = integer_at(edges, e+k);
        if (i < 0 || i >= n || j < 0 || j >= n) goto done;
        Py_ssize_t a = 0, b = 0;
        if (axis < 0) {
            a = root(parent, (Py_ssize_t)i);
            b = root(parent, (Py_ssize_t)j);
            if (a == b) continue;
        }
        if (cutoff2 >= 0) {
            double square = 0, scale = 0;
            for (int d = 0; d < 3; ++d) {
                double dx = real_at(x, 3*j+d) - real_at(x, 3*i+d);
                double translation = axis < 0 ? 0 : (double)h * cell[3*axis+d];
                double v = dx + translation, s = fabs(dx) + fabs(translation);
                square += v*v; scale += s*s;
            }
            /* Leave rounding-sensitive boundary edges to the NumPy reference. */
            if (!(square + 32*DBL_EPSILON*(scale + cutoff2) < cutoff2)) continue;
        }
        if (axis < 0) { parent[b] = a; --components; }
        else axes |= 1 << axis;
        if (components == 1 && axes == 7) { certified = 1; break; }
    }
done:
    PyMem_Free(parent);
    for (int k = 0; k < 4; ++k) if (buf[k].obj) PyBuffer_Release(&buf[k]);
    if (PyErr_Occurred()) return NULL;
    return PyBool_FromLong(certified);
}

static PyMethodDef methods[] = {{"bulk", bulk, METH_VARARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_connectivity", NULL, -1, methods};
PyMODINIT_FUNC PyInit__connectivity(void) { return PyModule_Create(&module); }
