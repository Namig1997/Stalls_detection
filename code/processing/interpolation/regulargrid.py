import numpy as np
from scipy.interpolate import RegularGridInterpolator


class InterpolatorRegularGrid:
    """
    Interpolator

    Class for changing of 3D grid size with interpolation

    Args:
        shape: tuple(3); (32, 32, 32);
            shape of output grid;
        method: str; "linear"; (see scipy RegularGridInterpolator);
        bounds_error: bool; False; (see scipy RegularGridInterpolator);
        fill_value: 0; float; (see scipy RegularGridInterpolator).
    """
    def __init__(self, 
            shape=(32, 32, 32),
            method="linear",
            bounds_error=False,
            fill_value=0,
            ):
        self.shape = shape
        self.method = method
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def get_coords(self, 
                shape, 
                size_z=1., 
                size_xy=1., 
                shift_z=0.,
                shift_x=0.,
                shift_y=0.,
                # rotation_xy=0,
                # rotation_phi=0,
                # rotation_psi=0,
                # rotation_theta=0,
                ):
        """
        Gets coordinates of grid vertices with respect to input volume shape.
        Args:
            shape: tuple(3); 
                shape of input volume.
        Returns:
            coords: np.array((self.shape[0]*self.shape[1]*self.shape[2], 3), np.float32);
                coordinates of grid vertices with respect to input shape.
        """
        # coords1 = np.linspace(0, shape[0]-1, self.shape[0])
        # coords2 = np.linspace(0, shape[1]-1, self.shape[1])
        # coords3 = np.linspace(0, shape[2]-1, self.shape[2])
        # return coords1, coords2, coords3
        # coords = np.empty((self.shape[0]*self.shape[1]*self.shape[2], 3), np.float32)
        # for i, c1 in enumerate(coords1):
        #     for j, c2 in enumerate(coords2):
        #         for k, c3 in enumerate(coords3):
        #             coords[i * self.shape[1] * self.shape[2] + j * self.shape[2] + k] = \
        #                 [c1, c2, c3]
        shift = np.array([
            shift_z * shape[0],
            shift_x * shape[1],
            shift_y * shape[2],
        ], np.float32)
        size = np.array([size_z, size_xy, size_xy], np.float32)
        coord_start = shift - 1. + size
        step = np.array([
            shape[0] / self.shape[0],
            shape[1] / self.shape[1],
            shape[2] / self.shape[2],
        ], np.float32) * size
        # print(shape)
        # print(shift)
        # print(coord_start)
        # print(step)
        
        coords = np.empty((self.shape[0]*self.shape[1]*self.shape[2], 3), np.float32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    index = i * self.shape[1] * self.shape[2] + j * self.shape[2] + k
                    indexes = np.array([i, j, k], np.float32)
                    coords[index] = indexes * step + coord_start
        return coords

    def assign(self, volume, grid, **kwargs):
        """
        Assigns interpolated from input 3D volume to grid.
        Args:
            volume: np.array((shape_x, shape_y, shape_z), np.float32);
                3D array to interpolate specified grid by;
            grid: np.array((self.shape[0], self.shape[1], self.shape[2]), np.float32);
                numpy array to store interpolated values into.
        """
        ind1 = np.arange(volume.shape[0])
        ind2 = np.arange(volume.shape[1])
        ind3 = np.arange(volume.shape[2])
        fn = RegularGridInterpolator(
            (ind1, ind2, ind3), volume, 
            method=self.method, 
            bounds_error=self.bounds_error, 
            fill_value=self.fill_value)
        grid[:] = np.reshape(fn(self.get_coords(volume.shape, **kwargs)), self.shape)
        # c1, c2, c3 = self.get_coords(volume.shape, **kwargs)
        # grid[:] = interp3D(volume, c1, c2, c3)
        return
        
    def interpolate(self, volume, **kwargs):
        """
        Calculates interpolation of 3D volume.
        """
        grid = np.empty(self.shape, np.float32)
        self.assign(volume, grid, **kwargs)
        return grid

    def __call__(self, volume, **kwargs):
        return self.interpolate(volume, **kwargs)







# # from https://stackoverflow.com/questions/41220617/python-3d-interpolation-speedup

# import numpy as np
# cimport numpy as np
# from libc.math cimport floor
# from cython cimport boundscheck, wraparound, nonecheck, cdivision

# DTYPE = np.float32
# ctypedef np.float32_t DTYPE_t

# @boundscheck(False)
# @wraparound(False)
# @nonecheck(False)
# def interp3D(DTYPE_t[:,:,::1] v, DTYPE_t[:,:,::1] xs, DTYPE_t[:,:,::1] ys, DTYPE_t[:,:,::1] zs):

#     cdef int X, Y, Z
#     X,Y,Z = v.shape[0], v.shape[1], v.shape[2]
#     cdef np.ndarray[DTYPE_t, ndim=3] interpolated = np.zeros((X, Y, Z), dtype=DTYPE)

#     _interp3D(&v[0,0,0], &xs[0,0,0], &ys[0,0,0], &zs[0,0,0], &interpolated[0,0,0], X, Y, Z)
#     return interpolated


# @cdivision(True)
# cdef inline void _interp3D(DTYPE_t *v, DTYPE_t *x_points, DTYPE_t *y_points, DTYPE_t *z_points, 
#                DTYPE_t *result, int X, int Y, int Z):

#     cdef:
#         int i, x0, x1, y0, y1, z0, z1, dim
#         DTYPE_t x, y, z, xd, yd, zd, c00, c01, c10, c11, c0, c1, c

#     dim = X*Y*Z

#     for i in range(dim):
#         x = x_points[i]
#         y = y_points[i]
#         z = z_points[i]

#         x0 = <int>floor(x)
#         x1 = x0 + 1
#         y0 = <int>floor(y)
#         y1 = y0 + 1
#         z0 = <int>floor(z)
#         z1 = z0 + 1

#         xd = (x-x0)/(x1-x0)
#         yd = (y-y0)/(y1-y0)
#         zd = (z-z0)/(z1-z0)

#         if x0 >= 0 and y0 >= 0 and z0 >= 0:
#             c00 = v[Y*Z*x0+Z*y0+z0]*(1-xd) + v[Y*Z*x1+Z*y0+z0]*xd
#             c01 = v[Y*Z*x0+Z*y0+z1]*(1-xd) + v[Y*Z*x1+Z*y0+z1]*xd
#             c10 = v[Y*Z*x0+Z*y1+z0]*(1-xd) + v[Y*Z*x1+Z*y1+z0]*xd
#             c11 = v[Y*Z*x0+Z*y1+z1]*(1-xd) + v[Y*Z*x1+Z*y1+z1]*xd

#             c0 = c00*(1-yd) + c10*yd
#             c1 = c01*(1-yd) + c11*yd

#             c = c0*(1-zd) + c1*zd

#         else:
#             c = 0

#         result[i] = c 