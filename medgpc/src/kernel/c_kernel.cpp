/*
-------------------------------------------------------------------------
This is the function file for top kernel class.
All other kernels can inherit these functions.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <omp.h>

#include "kernel/c_kernel.h"
#include "util/global_settings.h"

using namespace std;


c_kernel::c_kernel(){
    kernel_name = "c_kernel";
    kernel_hyp_num = 0;
    kernel_grad_thread = -1;
}
c_kernel::c_kernel(const vector<int> &input_param){
    kernel_name = "c_kernel";

    kernel_param = input_param;
    kernel_grad_thread = -1;
}
c_kernel::c_kernel(const vector<int> &input_param, const vector<double> &input_hyp){
    kernel_name = "c_kernel";

    kernel_param = input_param;
    kernel_hyp = input_hyp;
    kernel_grad_thread = -1;
}

void c_kernel::compute_squared_dist(
                                    const vector<float> &x, 
                                    const vector<float> &x2, 
                                    float *&dist_matrix
                                    ){
    int dim1, dim2;
    int i, j;

    dim1 = (x.size());
    dim2 = (x2.size());

    // use vectorization + openmp
    // omp_set_num_threads(GLOBAL_OMP_KERNEL_THREAD_NUM);
    #pragma omp parallel for simd private(i, j) firstprivate(dim1, dim2)
    for(int index = 0; index < (dim1*dim2); index++){
        i = floor(index/dim2);
        j = (index % dim2);
        dist_matrix[index] = pow((x[i] - x2[j]), 2.0);
    }

}

void c_kernel::compute_abs_dist(
                                const vector<float> &x, 
                                const vector<float> &x2, 
                                float *&dist_matrix
                                ){
    int dim1, dim2;
    int i, j;

    dim1 = (x.size());
    dim2 = (x2.size());

    // #pragma omp parallel for private(i, j) firstprivate(dim1, dim2)
    for(i = 0; i < dim1; i++){
        for(j = 0; j < dim2; j++){
            dist_matrix[ i * dim2 + j ] = fabs(x[i] - x2[j]);
        }
    }

}

void c_kernel::compute_sin_dist(
                                double period, 
                                const vector<float> &x, 
                                const vector<float> &x2, 
                                float *&dist_matrix
                                ){
    int dim1, dim2;
    int i, j;

    dim1 = (x.size());
    dim2 = (x2.size());

    // #pragma omp parallel for private(i, j) firstprivate(period, dim1, dim2)
    for(i = 0; i < dim1; i++){
        for(j = 0; j < dim2; j++){
            float temp_dist;
            temp_dist = fabs((x[i] - x2[j])/period);
            dist_matrix[ i * dim2 + j ] = sin(PI*temp_dist);
        }
    }
}

void c_kernel::compute_scale_abs_dist(
                                double lengthscale, 
                                const vector<float> &x, 
                                const vector<float> &x2, 
                                float *&dist_matrix
                                ){
    int dim1, dim2;
    int i, j;

    dim1 = (x.size());
    dim2 = (x2.size());

    // #pragma omp parallel for private(i, j) firstprivate(dim1, dim2)
    for(i = 0; i < dim1; i++){
        for(j = 0; j < dim2; j++){
            dist_matrix[ i * dim2 + j ] = fabs(x[i] - x2[j])/lengthscale;
        }
    }
}

void c_kernel::compute_scale_squared_dist(
                                            double lengthscale, 
                                            const vector<float> &x, 
                                            const vector<float> &x2, 
                                            float *&dist_matrix
                                            ){
    int dim1, dim2;
    int i, j;

    dim1 = (x.size());
    dim2 = (x2.size());

    #pragma omp parallel for simd private(i, j) firstprivate(lengthscale, dim1, dim2)
    for(int index = 0; index < (dim1*dim2); index++){
        i = floor(index/dim2);
        j = (index % dim2);
        float temp_dist = (x[i] - x2[j])/lengthscale;
        dist_matrix[index] = pow(temp_dist, 2.0);
    }

}
void c_kernel::compute_meta_map(
                                int dim, 
                                const vector<double> &coregional_matrix, 
                                const vector<int> &meta, 
                                const vector<int> &meta2, 
                                vector<float> &map_matrix
                                ){
    int n1, n2;
    n1 = int(meta.size());
    n2 = int(meta2.size());

    #pragma omp parallel for firstprivate(n1, n2, dim)
    for(int index = 0; index < (n1*n2); index++){
        int i = floor(index/n2);
        int j = (index % n2);
        int k = meta[i]*dim + meta2[j];
        map_matrix[index] = (float)(coregional_matrix[k]);
    }

}

void c_kernel::print_kernel(){
    cout << "current kernel object: " << kernel_name
         << "; # of hyperparameters: " << kernel_hyp_num << endl;
}
void c_kernel::set_kernel_grad_thread(int input_thread_num){
    kernel_grad_thread = input_thread_num;
}
int c_kernel::get_kernel_grad_thread(){
    return kernel_grad_thread;
}

int c_kernel::get_kernel_hyp_num(){
    return kernel_hyp_num;
}

vector<int> c_kernel::get_kernel_param(){
    vector<int> kernel_param_copy(kernel_param);
    return kernel_param_copy;
}

vector<double> c_kernel::get_kernel_hyp(){
    vector<double> kernel_hyp_copy(kernel_hyp);
    return kernel_hyp_copy;
}

