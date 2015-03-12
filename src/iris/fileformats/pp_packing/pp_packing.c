// (C) British Crown Copyright 2010 - 2012, Met Office
//
// This file is part of Iris.
//
// Iris is free software: you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Iris is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with Iris.  If not, see <http://www.gnu.org/licenses/>.
#include <Python.h>

#include <numpy/arrayobject.h>

#include <wgdosstuff.h>
#include <rlencode.h>

static PyObject *wgdos_unpack_py(PyObject *self, PyObject *args);
static PyObject *rle_decode_py(PyObject *self, PyObject *args);

#define BYTES_PER_INT_UNPACK_PPFIELD 4
#define LBPACK_WGDOS_PACKED 1
#define LBPACK_RLE_PACKED 4



void initpp_packing(void)
{

	/* The module doc string */
	PyDoc_STRVAR(pp_packing__doc__,
	"This extension module provides access to the underlying libmo_unpack library functionality.\n"
	""
	);

	PyDoc_STRVAR(wgdos_unpack__doc__,
	"Unpack PP field data that has been packed using WGDOS archive method.\n"
	"\n"
        "Provides access to the libmo_unpack library function Wgdos_Unpack.\n"
        "\n"
        "Args:\n\n"
        "* data (numpy.ndarray):\n"
        "    The raw field byte array to be unpacked.\n"
        "* lbrow (int):\n"
        "    The number of rows in the grid.\n"
        "* lbnpt (int):\n"
        "    The number of points (columns) per row in the grid.\n"
        "* bmdi (float):\n"
        "    The value used in the field to indicate missing data points.\n"
        "\n"
        "Returns:\n"
        "    numpy.ndarray, 2d array containing normal unpacked field data.\n" 
	""
	);


	PyDoc_STRVAR(rle_decode__doc__,
	"Uncompress PP field data that has been compressed using Run Length Encoding.\n"
	"\n"
        "Provides access to the libmo_unpack library function runlenDecode.\n"
        "Decodes the field by expanding out the missing data points represented\n"
        "by a single missing data value followed by a value indicating the length\n"
        "of the run of missing data values.\n"
        "\n"
        "Args:\n\n"
        "* data (numpy.ndarray):\n"
        "    The raw field byte array to be uncompressed.\n"
        "* lbrow (int):\n"
        "    The number of rows in the grid.\n"
        "* lbnpt (int):\n"
        "    The number of points (columns) per row in the grid.\n"
        "* bmdi (float):\n"
        "    The value used in the field to indicate missing data points.\n"
        "\n"
        "Returns:\n"
        "    numpy.ndarray, 2d array containing normal uncompressed field data.\n"
        ""
	);

	/* ==== Set up the module's methods table ====================== */
	static PyMethodDef pp_packingMethods[] = {
	    {"wgdos_unpack", wgdos_unpack_py, METH_VARARGS, wgdos_unpack__doc__},
	    {"rle_decode", rle_decode_py, METH_VARARGS, rle_decode__doc__},
	    {NULL, NULL, 0, NULL}     /* marks the end of this structure */
	};


       Py_InitModule3("pp_packing", pp_packingMethods, pp_packing__doc__);
       import_array();  // Must be present for NumPy.
}


/* wgdos_unpack(byte_array, lbrow, lbnpt, mdi) */
static PyObject *wgdos_unpack_py(PyObject *self, PyObject *args)
{
    char *bytes_in=NULL;
    PyArrayObject *npy_array_out=NULL;
    int bytes_in_len;
    npy_intp dims[2];
    int lbrow, lbnpt, npts;
    float mdi;

    if (!PyArg_ParseTuple(args, "s#iif", &bytes_in, &bytes_in_len, &lbrow, &lbnpt, &mdi)) return NULL;

    // Unpacking algorithm accepts an int - so assert that lbrow*lbnpt does not overflow 
    if (lbrow > 0 && lbnpt >= INT_MAX / (lbrow+1)) {
        PyErr_SetString(PyExc_ValueError, "Resulting unpacked PP field is larger than PP supports.");
        return NULL;
    } else{
        npts = lbnpt*lbrow;
    }

    // We can't use the macros Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS
    // because they declare a new scope block, but we want multiple exits.
    PyThreadState *_save;
    _save = PyEval_SaveThread();

    /* Do the unpack of the given byte array */
    float *dataout = (float*)calloc(npts, sizeof(float));

    if (dataout == NULL) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for wgdos_unpacking.");
        return NULL;
    }

    function func; // function is defined by wgdosstuff.
    set_function_name(__func__, &func, 0);
    int status = unpack_ppfield(mdi, 0, bytes_in, LBPACK_WGDOS_PACKED, npts, dataout, &func);

    /* Raise an exception if there was a problem with the WGDOS algorithm */
    if (status != 0) {
      free(dataout);
      PyEval_RestoreThread(_save);
      PyErr_SetString(PyExc_ValueError, "WGDOS unpack encountered an error."); 
      return NULL;
    }
    else {
        /* The data came back fine, so make a Numpy array and return it */
        dims[0]=lbrow;
        dims[1]=lbnpt;
        PyEval_RestoreThread(_save);
        npy_array_out=(PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, dataout);

        if (npy_array_out == NULL) {
          PyErr_SetString(PyExc_ValueError, "Failed to make the numpy array for the packed data.");
          return NULL;
        }

        // give ownership of dataout to the Numpy array - Numpy will then deal with memory cleanup.
        npy_array_out->flags = npy_array_out->flags | NPY_OWNDATA;

        return (PyObject *)npy_array_out;
    }
}


/* A null function required by the wgdos unpack library */
void MO_syslog(int value, char* message, const function* const caller)
{
	/* printf("MESSAGE %d %s: %s\n", value, caller, message); */
	return; 
}


/* rle_decode(byte_array, lbrow, lbnpt, mdi) */
static PyObject *rle_decode_py(PyObject *self, PyObject *args)
{
    char *bytes_in=NULL;
    PyArrayObject *npy_array_out=NULL;
    int bytes_in_len;
    npy_intp dims[2];
    int lbrow, lbnpt, npts;
    float mdi;

    if (!PyArg_ParseTuple(args, "s#iif", &bytes_in, &bytes_in_len, &lbrow, &lbnpt, &mdi)) return NULL;

    // Unpacking algorithm accepts an int - so assert that lbrow*lbnpt does not overflow
    if (lbrow > 0 && lbnpt >= INT_MAX / (lbrow+1)) {
	PyErr_SetString(PyExc_ValueError, "Resulting unpacked PP field is larger than PP supports.");
        return NULL;
    } else{
        npts = lbnpt*lbrow;
    }

    // We can't use the macros Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS
    // because they declare a new scope block, but we want multiple exits.
    PyThreadState *_save;
    _save = PyEval_SaveThread();

    float *dataout = (float*)calloc(npts, sizeof(float));

    if (dataout == NULL) {
        PyEval_RestoreThread(_save);
        PyErr_SetString(PyExc_ValueError, "Unable to allocate memory for wgdos_unpacking.");
        return NULL;
    }

    function func;  // function is defined by wgdosstuff.
    set_function_name(__func__, &func, 0);
    int status = unpack_ppfield(mdi, (bytes_in_len/BYTES_PER_INT_UNPACK_PPFIELD), bytes_in, LBPACK_RLE_PACKED, npts, dataout, &func);
    
    /* Raise an exception if there was a problem with the REL algorithm */
    if (status != 0) {
      free(dataout);
      PyEval_RestoreThread(_save);
      PyErr_SetString(PyExc_ValueError, "RLE decode encountered an error."); 
      return NULL;
    }
    else {
        /* The data came back fine, so make a Numpy array and return it */
        dims[0]=lbrow;
        dims[1]=lbnpt;
        PyEval_RestoreThread(_save);
        npy_array_out=(PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NPY_FLOAT, dataout);
        
        if (npy_array_out == NULL) {
          PyErr_SetString(PyExc_ValueError, "Failed to make the numpy array for the packed data.");
          return NULL;
        }

        // give ownership of dataout to the Numpy array - Numpy will then deal with memory cleanup.
        npy_array_out->flags = npy_array_out->flags | NPY_OWNDATA;
        return (PyObject *)npy_array_out;
   }
}
