/*
-------------------------------------------------------------------------
This is the function file for top kernel class.
All other kernels should inherit this class.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <mkl.h>
#include <omp.h>

#include "kernel/c_kernel_SE.h"
#include "util/global_settings.h"

using namespace std;


c_kernel_SE::c_kernel_SE(){
    kernel_name = "c_kernel_SE";
    kernel_hyp_num = 2;
}
c_kernel_SE::c_kernel_SE(const vector<int> &input_param){
    kernel_name = "c_kernel_SE";
    kernel_hyp_num = 2;
}
c_kernel_SE::c_kernel_SE(
                        const vector<int> &input_param, 
                        const vector<double> &input_hyp
                        ){
    kernel_name = "c_kernel_SE";
    kernel_hyp_num = 2;
    set_kernel_param(input_param);
    set_kernel_hyp(input_hyp);

    if(int(input_hyp.size()) != kernel_hyp_num){
        cout << "ERROR: mismatch # of hyperparameters! ";
        cout << "Get " << int(input_hyp.size()) << ", but expect " << kernel_hyp_num << endl;
        exit(0);
    }
}

void c_kernel_SE::set_kernel_param(const vector<int> &input_param){
    kernel_param = input_param;
}

void c_kernel_SE::set_kernel_hyp(const vector<double> &input_hyp){
    kernel_hyp = input_hyp;
    for(int i = 0; i < int(input_hyp.size()); i++){
        kernel_hyp[i] = exp(kernel_hyp[i]);
    }
}

void c_kernel_SE::compute_self_diag_matrix(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            float *&diag_gram_matrix
                                            ){
    int dim;
    int i;
    float diag_value;

    dim = int(x.size());
    diag_value = pow(kernel_hyp[1], 2.0);

    #pragma omp parallel for private(i) firstprivate(dim, diag_value)
    for(i = 0; i < dim; i++){
        diag_gram_matrix[i] = diag_value;
    }
}

void c_kernel_SE::compute_self_gram_matrix(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            float *&self_gram_matrix
                                            ){
    int dim = int(x.size());
    int i, j;

    compute_scale_squared_dist(kernel_hyp[0], x, x, self_gram_matrix);

    for(i = 0; i < dim; i++){
        #pragma omp parallel for private(j) firstprivate(dim, i)
        for(j = i; j < dim; j++){
            self_gram_matrix[ (i * dim) + j ] = pow(kernel_hyp[1], 2.0)*exp(-0.5*self_gram_matrix[ (i * dim) + j ]);
            self_gram_matrix[ (j * dim) + i ] = self_gram_matrix[ (i * dim) + j ];
        }
    }
}

void c_kernel_SE::compute_self_gradients(
                                        const vector<int> &meta, 
                                        const vector<float> &x,
                                        const float *chol_Q,
                                        vector<double> &gradients
                                        ){
    int     dim = int(x.size());
    int     i, j, index;
    float   *rsq, *map_matrix;
    double  derivative;

    map_matrix  = new float[dim*dim];
    rsq         = new float[dim*dim];
    compute_scale_squared_dist(kernel_hyp[0], x, x, rsq);
    
    gradients.clear();
    gradients.resize(kernel_hyp_num, 0.0);
    for(int hyp_index = 0; hyp_index < kernel_hyp_num; hyp_index++){
        if(hyp_index == 0){
            
            for(i = 0; i < dim; i++){
                #pragma omp parallel for private(j, index, derivative) firstprivate(dim, i)
                for(j = i; j < dim; j++){
                    index = (i * dim) + j;
                    derivative = pow(kernel_hyp[1], 2.)*exp(-0.5*rsq[index]);
                    derivative *= (rsq[index]);

                    map_matrix[index] = float(derivative);
                    map_matrix[ (j * dim) + i ] = float(derivative);
                }
            }
        }
        else if(hyp_index == 1){
            
            for(i = 0; i < dim; i++){
                #pragma omp parallel for private(j, index, derivative) firstprivate(dim, i)
                for(j = i; j < dim; j++){
                    index = (i * dim) + j;
                    derivative = 2.0*pow(kernel_hyp[1], 2.0)*exp(-0.5*rsq[index]);
                    
                    map_matrix[index] = float(derivative);
                    map_matrix[ (j * dim) + i ] = float(derivative);
                }
            }
        }
        gradients[hyp_index] = cblas_dsdot((dim*dim), chol_Q, 1, map_matrix, 1);
        gradients[hyp_index] = gradients[hyp_index]/2.0;
    }

    delete[] rsq;
    delete[] map_matrix;
}

void c_kernel_SE::compute_cross_gram_matrix(
                                            const vector<int> &meta, 
                                            const vector<int> &meta2, 
                                            const vector<float> &x, 
                                            const vector<float> &x2, 
                                            float *&cross_gram_matrix
                                            ){
    int dim1, dim2;
    int i, j;
    dim1 = int(x.size());
    dim2 = int(x2.size());
    
    compute_scale_squared_dist(kernel_hyp[0], x, x2, cross_gram_matrix);

    for(i = 0; i < dim1; i++){
        #pragma omp parallel for private(j) firstprivate(dim1, dim2, i)
        for(j = 0; j < dim2; j++){
            cross_gram_matrix[ (i * dim2) + j ] = pow(kernel_hyp[1], 2.)*exp(-0.5*cross_gram_matrix[(i * dim2) + j ]);
        }
    }
}


