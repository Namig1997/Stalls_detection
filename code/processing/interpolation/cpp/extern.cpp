#include "interpolation.hpp"


extern "C" {
    Interpolator*   interpolator_new    (const size_t* shape, 
                                        const size_t threads=1);
    
    void    interpolator_set_shape  (Interpolator* interpolator,
                                const size_t* shape);
    void    interpolator_set_threads(Interpolator* interpolator,
                                const size_t threads);
    void    interpolator_set_basis  (Interpolator* interpolator,
                                const float* basis);
    void    interpolator_set_center (Interpolator* interpolator,
                                const float* center);
    void    interpolator_set_origin (Interpolator* interpolator,
                                const float* origin);

    size_t* interpolator_get_shape  (Interpolator* interpolator);
    size_t  interpolator_get_threads(Interpolator* interpolator);
    float*  interpolator_get_basis  (Interpolator* interpolator);
    float*  interpolator_get_center (Interpolator* interpolator);
    float*  interpolator_get_origin (Interpolator* interpolator);

    float* interpolator_get_coordinate(Interpolator* interpolator,
                                        const size_t i, 
                                        const size_t j, 
                                        const size_t k);
    float interpolator_get_value(Interpolator* interpolator,
                                const float* coord, 
                                const float* grid, 
                                const size_t* shape);
    void interpolator_run(Interpolator* interpolator,
                            float* grid_out,
                            const float* grid_in, 
                            const size_t* shape_in);

    void interpolator_del(Interpolator* interpolator);
}


extern "C" {
    Interpolator* interpolator_new(const size_t* shape, const size_t threads) {
        return new Interpolator(shape, threads);
    }
    void interpolator_set_shape(Interpolator* interpolator, const size_t* shape){
        interpolator->set_shape(shape);
        return;
    }
    void interpolator_set_threads(Interpolator* interpolator, const size_t threads){
        interpolator->set_threads(threads);
        return;
    }
    void interpolator_set_basis(Interpolator* interpolator, const float* basis){
        interpolator->set_basis(basis);
        return;
    }
    void interpolator_set_center(Interpolator* interpolator, const float* center){
        interpolator->set_center(center);
        return;
    }
    void interpolator_set_origin(Interpolator* interpolator, const float* origin){
        interpolator->set_origin(origin);
        return;
    }
    size_t* interpolator_get_shape(Interpolator* interpolator){
        return interpolator->get_shape();
    }
    size_t interpolator_get_threads(Interpolator* interpolator){
        return interpolator->get_threads();
    }
    float* interpolator_get_basis(Interpolator* interpolator){
        return interpolator->get_basis();
    }
    float* interpolator_get_center(Interpolator* interpolator){
        return interpolator->get_center();
    }
    float* interpolator_get_origin(Interpolator* interpolator){
        return interpolator->get_origin();
    }
    float* interpolator_get_coordinate(Interpolator* interpolator,
            const size_t i, const size_t j, const size_t k){
        return interpolator->get_coordinate(i, j, k);
    }
    float interpolator_get_value(Interpolator* interpolator,
            const float* coord, const float* grid, const size_t* shape){
        return interpolator->get_value(coord, grid, shape);
    }
    void interpolator_run(Interpolator* interpolator,
            float* grid_out, const float* grid_in, const size_t* shape_in){
        interpolator->run(grid_out, grid_in, shape_in);
        return;
    }
    void interpolator_del(Interpolator* interpolator){
        delete interpolator;
        return;
    }
}