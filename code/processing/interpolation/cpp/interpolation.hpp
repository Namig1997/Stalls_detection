#pragma once

#include <cstddef>


class Interpolator {

private:
    size_t shape_[3] = {32, 32, 32};
    size_t threads_ = 1;

    float basis_[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    float origin_[3] = {0, 0, 0};

public:
    Interpolator(){};
    Interpolator(const size_t* shape, const size_t& threads=1);
    ~Interpolator(){};


    float* get_coordinate(
        const size_t& i, 
        const size_t& j, 
        const size_t& k) const;
    float get_value(
        const float* coord, 
        const float* grid, 
        const size_t* shape) const;
    void run(
        float* grid_out, 
        const float* grid_in, 
        const size_t* shape_in) const;
    void run_single(
        float* grid_out, 
        const float* grid_in, 
        const size_t* shape_in,
        const size_t* indexes_start,
        const size_t* indexes_end) const;

    void set_shape(const size_t* shape);
    void set_threads(const size_t& threads);
    void set_basis(const float* basis);
    void set_center(const float* center);
    void set_origin(const float* origin);

    size_t* get_shape() const;
    size_t get_threads() const;
    float* get_basis() const;
    float* get_center() const;
    float* get_origin() const;
};