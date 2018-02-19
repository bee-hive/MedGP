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

#include "c_kernel_SM.h"
#include "util/global_settings.h"

using namespace std;

c_kernel_SM::c_kernel_SM(){
    kernel_name =  "c_kernel_SM";
} 
c_kernel_SM::c_kernel_SM(const vector<int> &input_param){
    kernel_name =  "c_kernel_SM";
    set_kernel_param(input_param);
} 
c_kernel_SM::c_kernel_SM(
                        const vector<int> &input_param, 
                        const vector<double> &input_hyp
                        ){
    kernel_name =  "c_kernel_SM";
    set_kernel_param(input_param);
    set_kernel_hyp(input_hyp);

    if(int(input_hyp.size()) != kernel_hyp_num){
        cout << "ERROR: mismatch # of hyperparameters! ";
        cout << "Get " << int(input_hyp.size()) << ", but expect " << kernel_hyp_num << endl;
        exit(0);
    }
}

void c_kernel_SM::set_kernel_hyp(const vector<double> &input_hyp){
    kernel_hyp = input_hyp;
    for(int i = 0; i < int(input_hyp.size()); i++){
        kernel_hyp[i] = exp(kernel_hyp[i]);
    }
}

void c_kernel_SM::set_kernel_param(const vector<int> &input_param){
    kernel_param = input_param;
    int     Q = input_param[0];
    kernel_hyp_num = 3*Q;
}

void c_kernel_SM::compute_self_diag_matrix(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            float *&diag_gram_matrix
                                            ){
    int dim;
    int q, i;
    int Q = kernel_param[0];
    float diag_value;

    dim = int(x.size());
    diag_value = 0.0;
    for(q = 0; q < Q; q++){
        diag_value += kernel_hyp[q];
    }

    #pragma omp parallel for private(i) firstprivate(dim, diag_value)
    for(i = 0; i < dim; i++){
        diag_gram_matrix[i] = diag_value;
    }
}

void c_kernel_SM::compute_self_gram_matrix(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            float *&self_gram_matrix
                                            ){
    int dim = int(x.size());
    int q, i, j, index;
    int Q = kernel_param[0];
    double  temp_value;
    double  mu, v;

    float   *rsq;
    rsq     = new float[dim*dim];
    compute_squared_dist(x, x, rsq);

    for(q = 0; q < Q; q++){
        mu = kernel_hyp[q + Q];
        v  = kernel_hyp[q + 2*Q];
        for(i = 0; i < dim; i++){
            #pragma omp parallel for private(j) firstprivate(dim, mu, v, q, i)
            for(j = i; j < dim; j++){
                int index = (i * dim) + j;
                double curr_element = kernel_hyp[q]*compute_k(rsq[index], mu, v);
                if(q == 0){
                    self_gram_matrix[index] = (float)(curr_element);
                }
                else{
                    double sum = self_gram_matrix[index] + curr_element;
                    self_gram_matrix[index] = (float)(sum);
                }
                self_gram_matrix[ (j * dim) + i ] = self_gram_matrix[index];
            }
        }
    }
    delete[] rsq;
}

void c_kernel_SM::compute_self_gradients(
                                        const vector<int> &meta, 
                                        const vector<float> &x,
                                        const float *chol_Q,
                                        vector<double> &gradients
                                        ){
    int     dim = int(x.size());
    int     qidx, i, j, index;
    int     Q = kernel_param[0];
    float   *rsq, *map_matrix;
    double  derivative;

    rsq         = new float[dim*dim];
    map_matrix  = new float[dim*dim];

    compute_squared_dist(x, x, rsq);
    
    gradients.clear();
    gradients.resize(kernel_hyp_num, 0.0);
    for(int hyp_index = 0; hyp_index < kernel_hyp_num; hyp_index++){
        if(hyp_index < Q){
            qidx = hyp_index;
            #pragma omp parallel for private(i, j, index, derivative) firstprivate(dim, qidx)
            for(i = 0; i < dim; i++){
                for(j = i; j < dim; j++){
                    index = (i * dim) + j;
                    derivative = kernel_hyp[qidx]*compute_k(rsq[index], kernel_hyp[qidx + Q], kernel_hyp[qidx + 2*Q]);

                    map_matrix[index] = float(derivative);
                    map_matrix[ (j * dim) + i ] = float(derivative);
                }
            }
        }
        else if(hyp_index < 2*Q){
            qidx = hyp_index - Q;

            #pragma omp parallel for private(i, j, index, derivative) firstprivate(dim, qidx)
            for(i = 0; i < dim; i++){
                for(j = i; j < dim; j++){
                    index = (i * dim) + j;
                    derivative = kernel_hyp[qidx]*compute_km(rsq[index], kernel_hyp[qidx + Q], kernel_hyp[qidx + 2*Q]);

                    map_matrix[index] = float(derivative);
                    map_matrix[ (j * dim) + i ] = float(derivative);
                }
            }
        }
        else if(hyp_index < 3*Q){
            qidx = hyp_index - 2*Q;

            #pragma omp parallel for private(i, j, index, derivative) firstprivate(dim, qidx)
            for(i = 0; i < dim; i++){
                for(j = i; j < dim; j++){
                    index = (i * dim) + j;
                    derivative = kernel_hyp[qidx]*compute_kv(rsq[index], kernel_hyp[qidx + Q], kernel_hyp[qidx + 2*Q]);

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

void c_kernel_SM::compute_cross_gram_matrix(
                                            const vector<int> &meta, 
                                            const vector<int> &meta2, 
                                            const vector<float> &x, 
                                            const vector<float> &x2, 
                                            float *&cross_gram_matrix
                                            ){
    int     dim1, dim2;
    int     q, i, j, index;
    int     Q = kernel_param[0];
    double  temp_value;

    dim1 = int(x.size());
    dim2 = int(x2.size());
    
    double  mu, v;
    float   *rsq;
    rsq     = new float[dim1*dim2];
    compute_squared_dist(x, x2, rsq);

    for(q = 0; q < Q; q++){
        mu = kernel_hyp[q + Q];
        v  = kernel_hyp[q + 2*Q];
            
        for(i = 0; i < dim1; i++){
            #pragma omp parallel for private(j) firstprivate(dim1, dim2, mu, v, q, i)
            for(j = 0; j < dim2; j++){
                int index = (i * dim2) + j;
                double curr_element = kernel_hyp[q]*compute_k(rsq[index], mu, v);
                if(q == 0){
                    cross_gram_matrix[index] = (float)(curr_element);
                }
                else{
                    double sum = cross_gram_matrix[index] + curr_element;
                    cross_gram_matrix[index] = (float)(sum);
                }
            }
        }
    }
    delete[] rsq;
}

float c_kernel_SM::compute_k(const float &rsq, const double &mu, const double &v){
    double value;
    value = cos(2.0*PI*sqrt(rsq)*mu)*exp(-2.0*pow(PI*v, 2.0)*rsq);
    return (float) value;
}
float c_kernel_SM::compute_km(const float &rsq, const double &mu, const double &v){
    double value;
    double dmu = 2.0*PI*sqrt(rsq)*mu;
    value = (-1.0)*dmu*sin(dmu)*exp(-2.0*pow(PI*v, 2.0)*rsq);
    return (float) value;
}
float c_kernel_SM::compute_kv(const float &rsq, const double &mu, const double &v){
    double value;
    double d2piv = pow(PI*v, 2.0)*rsq;
    value = cos(2.0*PI*sqrt(rsq)*mu)*exp(-2.0*d2piv);
    value *= -4.0*(d2piv);
    return (float) value;
}

