import os
import sys
import numpy as np
from ctypes import cdll
from ctypes import c_void_p, c_size_t, c_float
_folder_current = os.path.dirname(os.path.abspath(__file__))
_filename_lib   = os.path.join(_folder_current, "cpp", "libinterpolation.so")
lib = cdll.LoadLibrary(_filename_lib)


class InterpolatorCustom:
    def __init__(self, 
            shape=(32, 32, 32),
            threads=8):
        lib.interpolator_new.argtypes = [
            np.ctypeslib.ndpointer(dtype=c_size_t, shape=(3,)),
            c_size_t,]
        lib.interpolator_new.restype = c_void_p

        lib.interpolator_set_shape.argtypes = [
            c_void_p, np.ctypeslib.ndpointer(dtype=c_size_t, shape=(3,))]
        lib.interpolator_set_shape.restype = c_void_p
        lib.interpolator_set_threads.argtypes = [
            c_void_p, c_size_t,]
        lib.interpolator_set_threads.restype = c_void_p
        lib.interpolator_set_basis.argtypes = [
            c_void_p, np.ctypeslib.ndpointer(dtype=c_float, shape=(3, 3,))]
        lib.interpolator_set_basis.restype = c_void_p
        lib.interpolator_set_center.argtypes = [
            c_void_p, np.ctypeslib.ndpointer(dtype=c_float, shape=(3,))]
        lib.interpolator_set_center.restype = c_void_p
        lib.interpolator_set_origin.argtypes = [
            c_void_p, np.ctypeslib.ndpointer(dtype=c_float, shape=(3,))]
        lib.interpolator_set_origin.restype = c_void_p

        lib.interpolator_get_shape.argtypes    = [c_void_p]
        lib.interpolator_get_shape.restype     = \
            np.ctypeslib.ndpointer(dtype=c_size_t, shape=(3,))
        lib.interpolator_get_threads.argtypes  = [c_void_p]
        lib.interpolator_get_threads.restype   = c_size_t
        lib.interpolator_get_basis.argtypes    = [c_void_p]
        lib.interpolator_get_basis.restype     = \
            np.ctypeslib.ndpointer(dtype=c_float, shape=(3, 3,))
        lib.interpolator_get_center.argtypes   = [c_void_p]
        lib.interpolator_get_center.restype    = \
            np.ctypeslib.ndpointer(dtype=c_float, shape=(3,))
        lib.interpolator_get_origin.argtypes   = [c_void_p]
        lib.interpolator_get_origin.restype    = \
            np.ctypeslib.ndpointer(dtype=c_float, shape=(3,))

        lib.interpolator_get_coordinate.argtypes = [c_void_p,
            c_size_t, c_size_t, c_size_t]
        lib.interpolator_get_coordinate.restype = \
            np.ctypeslib.ndpointer(dtype=c_float, shape=(3,))
        lib.interpolator_get_value.restype = c_float
        lib.interpolator_run.restype = c_void_p
        lib.interpolator_del.argtypes = [c_void_p]
        lib.interpolator_del.restype = c_void_p

        shape = np.array(shape, dtype=c_size_t)
        self.obj = lib.interpolator_new(shape, threads)

    @property
    def shape(self):
        return lib.interpolator_get_shape(self.obj).copy()
    @property
    def threads(self):
        return lib.interpolator_get_threads(self.obj)
    @property
    def basis(self):
        return lib.interpolator_get_basis(self.obj).copy()
    @property
    def center(self):
        return lib.interpolator_get_center(self.obj).copy()
    @property
    def origin(self):
        return lib.interpolator_get_origin(self.obj).copy()

    @shape.setter
    def shape(self, shape):
        shape = np.array(shape, dtype=c_size_t)
        lib.interpolator_set_shape(self.obj, shape)
    @threads.setter
    def threads(self, threads):
        lib.interpolator_set_threads(self.obj, threads)
    @basis.setter
    def basis(self, basis):
        basis = np.array(basis, dtype=c_float)
        lib.interpolator_set_basis(self.obj, basis)
    @center.setter
    def center(self, center):
        center = np.array(center, dtype=c_float)
        lib.interpolator_set_center(self.obj, center)
    @origin.setter
    def origin(self, origin):
        origin = np.array(origin, dtype=c_float)
        lib.interpolator_set_origin(self.obj, origin)

    def set_basis(self, 
            shape, 
            size_z=1.,
            size_xy=1.,
            shift_z=0.,
            shift_x=0.,
            shift_y=0.,
            rotation_theta=0,
            rotation_phi=0,
            rotation_psi=0,
            ):
        center = (np.array(shape, np.float32) - 1.) / 2.
        center[0] += shift_z * shape[0]
        center[1] += shift_x * shape[1]
        center[2] += shift_y * shape[2]
        basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], np.float32)
        rotation_matrix = self.rotation_matrix(rotation_theta, rotation_phi, rotation_psi)
        basis = np.dot(basis, rotation_matrix)
        basis[:, 0] *= (shape[0] - 1) / self.shape[0] * size_z
        basis[:, 1] *= (shape[1] - 1) / self.shape[1] * size_xy
        basis[:, 2] *= (shape[2] - 1) / self.shape[2] * size_xy
        self.basis = basis
        self.center = center

    def get_coordinate(self, i, j, k):
        return lib.interpolator_get_coordinate(self.obj, i, j, k).copy()

    def get_value(self, coord, grid):
        shape = np.array(grid.shape, dtype=c_size_t)
        lib.interpolator_get_value.argtypes = [c_void_p,
            np.ctypeslib.ndpointer(dtype=c_float, shape=(3,)),
            np.ctypeslib.ndpointer(dtype=c_float, shape=shape),
            np.ctypeslib.ndpointer(dtype=c_size_t, shape=(3,)),
        ]
        return lib.interpolator_get_value(self.obj, coord, grid, shape)

    def run(self, grid_out, grid_in):
        shape = np.array(grid_in.shape, dtype=c_size_t)
        lib.interpolator_run.argtypes = [c_void_p,
            np.ctypeslib.ndpointer(dtype=c_float, shape=self.shape),
            np.ctypeslib.ndpointer(dtype=c_float, shape=shape),
            np.ctypeslib.ndpointer(dtype=c_size_t, shape=(3,))
        ]
        lib.interpolator_run(self.obj, grid_out, grid_in, shape)

    def interpolate(self, grid, **kwargs):
        grid_out = np.empty(self.shape, np.float32)
        self.set_basis(grid.shape, **kwargs)
        self.run(grid_out, grid)
        return grid_out

    def assign(self, volume, grid, **kwargs):
        self.set_basis(volume.shape, **kwargs)
        self.run(grid, volume)
        return

    def __call__(self, grid, **kwargs):
        return self.interpolate(grid, **kwargs)

    def __del__(self):
        lib.interpolator_del(self.obj)


    @staticmethod
    def rotation_matrix(theta, phi, psi):
        """
        Calculates rotation matrix from given angles.
        Args:
            theta:  float; for axis to rotate around;
            phi:    float; for axis to rotate around;
            psi:    float; angle to rotate by.
        Returns:
            np.array((3, 3), np.float32); rotation matrix.
        """
        u = np.array([
            np.cos(phi) * np.sin(theta), 
            np.sin(phi) * np.sin(theta),
            np.cos(theta)], dtype=np.float32)
        c, s, C = np.cos(psi), np.sin(psi), 1 - np.cos(psi)
        matrix = np.array([
            [u[0]*u[0]*C + c,      u[0]*u[1]*C - u[2]*s, u[0]*u[2]*C + u[1]*s],
            [u[1]*u[0]*C + u[2]*s, u[1]*u[1]*C + c,      u[1]*u[2]*C - u[0]*s],
            [u[2]*u[0]*C - u[1]*s, u[1]*u[2]*C + u[0]*s, u[2]*u[2]*C + c],
        ], dtype=np.float32)
        return matrix