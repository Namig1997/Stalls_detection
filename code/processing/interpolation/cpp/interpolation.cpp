#include "interpolation.hpp"
#include <algorithm>
#include <math.h>
#include <future>
#include <vector>
#include <iostream>

Interpolator::Interpolator(const size_t* shape, const size_t& threads){
    threads_ = threads;
    set_shape(shape);
}


void Interpolator::set_shape(const size_t* shape){
    shape_[0] = shape[0];
    shape_[1] = shape[1];
    shape_[2] = shape[2];
}
void Interpolator::set_threads(const size_t& threads){
    threads_ = threads;
}
void Interpolator::set_basis(const float* basis){
    basis_[0][0] = basis[0];
    basis_[0][1] = basis[1];
    basis_[0][2] = basis[2];
    basis_[1][0] = basis[3];
    basis_[1][1] = basis[4];
    basis_[1][2] = basis[5];
    basis_[2][0] = basis[6];
    basis_[2][1] = basis[7];
    basis_[2][2] = basis[8];
}
void Interpolator::set_center(const float* center){
    origin_[0] = center[0] - static_cast<float>(shape_[0]) / 2. * 
        (basis_[0][0] + basis_[1][0] + basis_[2][0]);
    origin_[1] = center[1] - static_cast<float>(shape_[1]) / 2. *
        (basis_[0][1] + basis_[1][1] + basis_[2][1]);
    origin_[2] = center[2] - static_cast<float>(shape_[2]) / 2. *
        (basis_[0][2] + basis_[1][2] + basis_[2][2]);
}
void Interpolator::set_origin(const float* origin){
    origin_[0] = origin[0];
    origin_[1] = origin[1];
    origin_[2] = origin[2];
}

size_t* Interpolator::get_shape() const {
    static size_t shape[3];
    shape[0] = shape_[0];
    shape[1] = shape_[1];
    shape[2] = shape_[2];
    return shape;
}
size_t Interpolator::get_threads() const {
    return threads_;
}
float* Interpolator::get_basis() const {
    static float basis[9];
    basis[0] = basis_[0][0];
    basis[1] = basis_[0][1];
    basis[2] = basis_[0][2];
    basis[3] = basis_[1][0];
    basis[4] = basis_[1][1];
    basis[5] = basis_[1][2];
    basis[6] = basis_[2][0];
    basis[7] = basis_[2][1];
    basis[8] = basis_[2][2];
    return basis;
}
float* Interpolator::get_center() const {
    static float center[3];
    center[0] = origin_[0] + static_cast<float>(shape_[0]) / 2. * 
        (basis_[0][0] + basis_[1][0] + basis_[2][0]);
    center[1] = origin_[1] + static_cast<float>(shape_[1]) / 2. * 
        (basis_[0][1] + basis_[1][1] + basis_[2][1]);
    center[2] = origin_[2] + static_cast<float>(shape_[2]) / 2. * 
        (basis_[0][2] + basis_[1][2] + basis_[2][2]);
    return center;
}
float* Interpolator::get_origin() const {
    static float origin[3];
    origin[0] = origin_[0];
    origin[1] = origin_[1];
    origin[2] = origin_[2];
    return origin;
}


float* Interpolator::get_coordinate(
        const size_t& i, 
        const size_t& j, 
        const size_t& k) const {
    static float coord[3];
    float i_float = static_cast<float>(i);
    float j_float = static_cast<float>(j);
    float k_float = static_cast<float>(k);
    coord[0] = origin_[0] + 
        i_float * basis_[0][0] +
        j_float * basis_[1][0] + 
        k_float * basis_[2][0];
    coord[1] = origin_[1] + 
        i_float * basis_[0][1] +
        j_float * basis_[1][1] +
        k_float * basis_[2][1];
    coord[2] = origin_[2] +
        i_float * basis_[0][2] +
        j_float * basis_[1][2] +
        k_float * basis_[2][2];
    return coord;
}

float Interpolator::get_value(
        const float* coord, 
        const float* grid, 
        const size_t* shape) const {

    // https://en.wikipedia.org/wiki/Trilinear_interpolation
    
    int i0, i1, j0, j1, k0, k1;
    i0 = static_cast<int>(std::floor(coord[0]));
    i1 = i0 + 1;
    j0 = static_cast<int>(std::floor(coord[1]));
    j1 = j0 + 1;
    k0 = static_cast<int>(std::floor(coord[2]));
    k1 = k0 + 1;

    if ((i1 < 0) || (i0 >= shape[0]) || 
        (j1 < 0) || (j0 >= shape[1]) || 
        (k1 < 0) || (k0 >= shape[2])) {
        return 0;
    }

    float c000, c001, c010, c011, c100, c101, c110, c111;
    int i0_shape = i0*shape[1]*shape[2];
    int i1_shape = i1*shape[1]*shape[2];
    int j0_shape = j0*shape[2];
    int j1_shape = j1*shape[2];

    if ((i0 < 0) || (j0 < 0) || (k0 < 0)) {
        c000 = 0;
    } else {
        // c000 = grid[i0*shape[1]*shape[2] + j0*shape[2] + k0];
        c000 = grid[i0_shape + j0_shape + k0];
    }
    if ((i0 < 0) || (j0 < 0) || (k1 >= shape[2])) {
        c001 = 0;
    } else {
        // c001 = grid[i0*shape[1]*shape[2] + j0*shape[2] + k1];
        c001 = grid[i0_shape + j0_shape + k1];
    }
    if ((i0 < 0) || (j1 >= shape[1]) || (k0 < 0)) {
        c010 = 0;
    } else {
        // c010 = grid[i0*shape[1]*shape[2] + j1*shape[2] + k0];
        c010 = grid[i0_shape + j1_shape + k0];
    }
    if ((i0 < 0) || (j1 >= shape[1]) || (k1 >= shape[2])) {
        c011 = 0;
    } else {
        // c011 = grid[i0*shape[1]*shape[2] + j1*shape[2] + k1];
        c011 = grid[i0_shape + j1_shape + k1];
    }
    if ((i1 >= shape[0]) || (j0 < 0) || (k0 < 0)) {
        c100 = 0;
    } else {
        // c100 = grid[i1*shape[1]*shape[2] + j0*shape[2] + k0];
        c100 = grid[i1_shape + j0_shape + k0];
    }
    if ((i1 >= shape[0]) || (j0 < 0) || (k1 >= shape[2])) {
        c101 = 0;
    } else {
        // c101 = grid[i1*shape[1]*shape[2] + j0*shape[2] + k1];
        c101 = grid[i1_shape + j0_shape + k1];
    }
    if ((i1 >= shape[0]) || (j1 >= shape[1]) || (k0 < 0)) {
        c110 = 0;
    } else {
        // c110 = grid[i1*shape[1]*shape[2] + j1*shape[2] + k0];
        c110 = grid[i1_shape + j1_shape + k0];
    }
    if ((i1 >= shape[0]) || (j1 >= shape[1]) || (k1 >= shape[2])) {
        c111 = 0;
    } else {
        // c111 = grid[i1*shape[1]*shape[2] + j1*shape[2] + k1];
        c111 = grid[i1_shape + j1_shape + k1];
    }

    float xd = coord[0] - i0;
    float yd = coord[1] - j0;
    float zd = coord[2] - k0;
    float xd1 = 1. - xd;
    float yd1 = 1. - yd;
    float zd1 = 1. - zd;

    float c00 = c000*xd1 + c100*xd;
    float c01 = c001*xd1 + c101*xd;
    float c10 = c010*xd1 + c110*xd;
    float c11 = c011*xd1 + c111*xd;
    float c0 = c00*yd1 + c10*yd;
    float c1 = c01*yd1 + c11*yd;
    return c0*zd1 + c1*zd; 
}

void Interpolator::run_single(
        float* grid_out,
        const float* grid_in,
        const size_t* shape_in,
        const size_t* indexes_start,
        const size_t* indexes_end) const {

    size_t index;
    for (size_t i=indexes_start[0]; i<std::min(indexes_end[0], shape_[0]); ++i){
        for (size_t j=indexes_start[1]; j<std::min(indexes_end[1], shape_[1]); ++j) {
            for (size_t k=indexes_start[2]; k<std::min(indexes_end[2], shape_[2]); ++k){
                index = i*shape_[1]*shape_[2] + j*shape_[2] + k;
                grid_out[index] = get_value(
                    get_coordinate(i, j, k),
                    grid_in, shape_in);
            }
        }
    }
    return;
}

void Interpolator::run(
        float* grid_out, 
        const float* grid_in,
        const size_t* shape_in) const {
    // size_t index;
    // for(size_t i=0; i<shape_[0]; ++i){
    //     for (size_t j=0; j<shape_[1]; ++j){
    //         for (size_t k=0; k<shape_[2]; ++k){
    //             index = i*shape_[1]*shape_[2] + j*shape_[2] + k;
    //             grid_out[index] = get_value(get_coordinate(i, j, k), 
    //                 grid_in, shape_in);
    //         }
    //     }
    // }
    if (threads_ == 1) {
        size_t indexes_start[3] = {0, 0, 0};
        run_single(grid_out, grid_in, shape_in, indexes_start, shape_);
    } else {
        std::vector<std::future<void>> futures;
        size_t chunk_size = std::ceil(static_cast<float>(shape_[0]) / threads_);
        for (size_t thread=0; thread<threads_; ++thread){
            futures.push_back(
                std::async([this, grid_out, grid_in, shape_in, chunk_size, thread]{
                    size_t indexes_start[3] = {thread*chunk_size, 0, 0};
                    size_t indexes_end[3] = {chunk_size*(thread+1), shape_[1], shape_[2]};
                    run_single(grid_out, grid_in, shape_in, indexes_start, indexes_end);
                })
            );
        }
    }
}