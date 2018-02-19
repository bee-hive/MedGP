/*
-------------------------------------------------------------------------
This is the function file for linear model of coregional + spectral kernel.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <time.h>

#include "kernel/c_kernel_LMC_SM.h"
#include "util/global_settings.h"

using namespace std;


c_kernel_LMC_SM::c_kernel_LMC_SM(){
    kernel_name = "c_kernel_LMC_SM";
    kernel_grad_thread = -1;
}
c_kernel_LMC_SM::c_kernel_LMC_SM(const vector<int> &input_param){
    kernel_name = "c_kernel_LMC_SM";
    kernel_grad_thread = -1;
    set_kernel_param(input_param);
}

c_kernel_LMC_SM::c_kernel_LMC_SM(
                                const vector<int> &input_param, 
                                const vector<double> &input_hyp
                                ){
    kernel_name = "c_kernel_LMC_SM";
    kernel_grad_thread = -1;
    set_kernel_param(input_param);
    set_kernel_hyp(input_hyp);

    if(int(input_param.size()) != 3){
        cout << "ERROR:current input parameters should report 3 numbers (mixture, output, rank); " 
             << "received " << input_param.size() << endl;
        exit(1);
    }

    if(int(input_hyp.size()) != kernel_hyp_num){
        cout << "ERROR: mismatch # of hyperparameters! ";
        cout << "Get " << int(input_hyp.size()) << ", but expect " << kernel_hyp_num << endl;
        exit(1);
    }
}

void c_kernel_LMC_SM::set_kernel_hyp(const vector<double> &input_hyp){
    int Q = kernel_param[0];
    int D = kernel_param[1];
    int R = kernel_param[2];

    kernel_hyp = input_hyp;
    for(int i = Q*D*R; i < int(input_hyp.size()); i++){
        kernel_hyp[i] = exp(kernel_hyp[i]);
    }

    compute_coregional_matrix();
}

void c_kernel_LMC_SM::set_kernel_param(const vector<int> &input_param){
    kernel_param = input_param;
    int     Q = input_param[0];
    int     D = input_param[1];
    int     R = input_param[2];
    kernel_hyp_num = Q*(D*R + 2 + D);
}

void c_kernel_LMC_SM::compute_coregional_matrix(){
    int     q, i, j;
    int     a_offset, kappa_offset;

    int     Q = kernel_param[0];
    int     D = kernel_param[1];
    int     R = kernel_param[2];
    
    double  *a_matrix, *b_matrix;
    a_matrix = new double[D*R];
    b_matrix = new double[D*D];

    // clear possible previous matrix
    coregional_matrix.clear(); 
    for(q = 0; q < Q; q++){
        vector<double>  comp_matrix;
        a_offset = q*D*R;
        kappa_offset = Q*(D*R + 2) + q*D;

        // copy a matrix out
        for(i = 0; i < D; i++){
            for(j = 0; j < R; j++){
                a_matrix[ i*R + j ] = kernel_hyp[ a_offset + i*R + j ];
            }
        }

        // compute coregional matrix for component q
        MKL_INT n1, n2;
        n1 = D;
        n2 = R;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n1, n1, n2, 1.0, a_matrix, n2, a_matrix, n2, 0.0, b_matrix, n1);

        // add kappa
        for(i = 0; i < D; i++){
            b_matrix[ i*D + i ] += kernel_hyp[kappa_offset + i];
        }

        // copy into vector
        comp_matrix.assign(b_matrix, b_matrix + D*D);
        coregional_matrix.push_back(comp_matrix);
    }
    delete[] a_matrix;
    delete[] b_matrix;
}

void c_kernel_LMC_SM::reset_coregional_matrix(vector< vector<double> > em_B_array){
    coregional_matrix.clear(); 
    coregional_matrix = em_B_array;
}

void c_kernel_LMC_SM::compute_self_diag_matrix(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            float *&diag_gram_matrix
                                            ){
    int     i, q;
    float   diag_value;

    int     Q = kernel_param[0];
    int     D = kernel_param[1];
    int     R = kernel_param[2];
    int     dim  = int(x.size());

    vector<float>   diag_vec(D, 0.0f);

    for(i = 0; i < D; i++){
        diag_value  = 0.0;
        #pragma omp parallel for private(q) firstprivate(i, Q, D) reduction(+: diag_value)
        for(q = 0; q < Q; q++){
            diag_value += coregional_matrix[q][ i*D + i ];
        }
        diag_vec[i] = diag_value;
    }

    #pragma omp parallel for private(i) firstprivate(dim)
    for(i = 0; i < dim; i++){
        diag_gram_matrix[i] = diag_vec[ meta[i] ];
    }
}

void c_kernel_LMC_SM::compute_self_gram_matrix(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            float *&self_gram_matrix
                                            ){
    int     dim = int(x.size());
    MKL_INT n = dim*dim;

    int     q, i, j;
    int     Q = kernel_param[0];
    int     D = kernel_param[1];
    int     R = kernel_param[2];

    float   temp_k_value;
    double  mu, v, s1, s2;
    time_t  t1, t2, t3, t4;

    float   *rsq;
    rsq         = new float[dim*dim];

    compute_squared_dist(x, x, rsq);

    for(q = 0; q < Q; q++){
        mu = kernel_hyp[Q*D*R + q];
        v  = kernel_hyp[Q*(D*R + 1) + q];

        for(i = 0; i < dim; i++){
            #pragma omp parallel for private(j) firstprivate(dim, mu, v, q, i)
            for(j = i; j < dim; j++){
                int index = (i * dim) + j;
                float curr_b_coef = coregional_matrix[q][meta[i]*D + meta[j]];
                double curr_element = curr_b_coef*compute_k(rsq[index], mu, v);
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

void c_kernel_LMC_SM::compute_self_gradients(
                                        const vector<int> &meta, 
                                        const vector<float> &x,
                                        const float *chol_Q,
                                        vector<double> &gradients
                                        ){
    int     dim = int(x.size());
    float   *rsq;
    
    time_t  t1, t2;
    int     Q = kernel_param[0];
    int     D = kernel_param[1];
    int     R = kernel_param[2];

    rsq             = new float[dim*dim];

    compute_squared_dist(x, x, rsq);

    gradients.clear();
    gradients.resize(kernel_hyp_num, 0.0);

    if(kernel_grad_thread > 0){
        omp_set_num_threads(kernel_grad_thread);
    }
    #pragma omp parallel for default(shared) firstprivate(dim, Q, D, R)
    for(int hyp_index = 0; hyp_index < kernel_hyp_num; hyp_index++){
        vector<float>   map_matrix(dim*dim, 0.0);
        omp_set_num_threads(1);
        mkl_set_num_threads_local(1);

        if(hyp_index < Q*D*R){ // a
            int offset  = Q*D*R;
            int qidx    = floor(hyp_index/(D*R));
            int mod_idx = (hyp_index % (D*R));
            int didx    = floor(mod_idx/R);
            int ridx    = (mod_idx % R);
            double mu       = kernel_hyp[offset + qidx];
            double v        = kernel_hyp[offset + Q + qidx];
            
            // compute partial of coregional matrix
            vector<double> sub_map_matrix(D*D, 0.0);
            for(int i = 0; i < D; i++){
                sub_map_matrix[ i*D + didx ] += kernel_hyp[qidx*D*R + i*R + ridx];
            }
            for(int i = 0; i < D; i++){
                sub_map_matrix[ didx*D + i ] += kernel_hyp[qidx*D*R + i*R + ridx];
            }

            compute_meta_map(D, sub_map_matrix, meta, meta, map_matrix);
            
            for(int i = 0; i < dim; i++){
                for(int j = i; j < dim; j++){
                    int index =  (i * dim) + j; 
                    double derivative = map_matrix[index] * compute_k(rsq[index], mu, v);
                    map_matrix[index] = (float)(derivative);
                    map_matrix[ (j * dim) + i ] = (float)(derivative);
                }
            }
        }
        else if(hyp_index < Q*(D*R + 1)){ // mu
            int offset  = Q*D*R;
            int qidx    = (hyp_index - offset) % Q;
            double mu       = kernel_hyp[offset + qidx];
            double v        = kernel_hyp[offset + Q + qidx];

            compute_meta_map(D, coregional_matrix[qidx], meta, meta, map_matrix);

            for(int i = 0; i < dim; i++){
                for(int j = i; j < dim; j++){
                    int index = (i * dim) + j;
                    double derivative = map_matrix[index] * compute_km(rsq[index], mu, v);
                    
                    map_matrix[index] = (float)(derivative);
                    map_matrix[ (j * dim) + i ] = (float)(derivative);
                }
            }

        }
        else if(hyp_index < Q*(D*R+2)){ // v
            int offset  = Q*( D*R + 1 );
            int qidx    = (hyp_index - offset) % Q;
            double mu       = kernel_hyp[offset - Q + qidx];
            double v        = kernel_hyp[offset + qidx];

            compute_meta_map(D, coregional_matrix[qidx], meta, meta, map_matrix);

            for(int i = 0; i < dim; i++){
                for(int j = i; j < dim; j++){
                    int index = (i * dim) + j;
                    double derivative = map_matrix[index] * compute_kv(rsq[index], mu, v);
                    
                    map_matrix[index] = (float)(derivative);
                    map_matrix[ (j * dim) + i ] = (float)(derivative);
                }
            }
        }
        else if(hyp_index < Q*(D*R + 2 + D)){ // kappa
            int offset  = Q*( D*R + 2 );
            int qidx    = floor((hyp_index - offset)/D);
            int didx    = (hyp_index-offset) % D;

            double mu       = kernel_hyp[offset - 2*Q + qidx];
            double v        = kernel_hyp[offset - Q + qidx];
            double s1       = 2.0*PI*mu;
            double s2       = -2.0*pow(PI*v, 2.0);

            // standard implementation
            // compute partial of coregional matrix
            vector<double> sub_map_matrix(D*D, 0.0);
            sub_map_matrix[ didx*D + didx ] = kernel_hyp[offset + qidx*D + didx];
            
            compute_meta_map(D, sub_map_matrix, meta, meta, map_matrix);
            
            for(int i = 0; i < dim; i++){
                for(int j = i; j < dim; j++){
                    int index = (i * dim) + j;
                    double derivative = map_matrix[index] * compute_k(rsq[index], mu, v);
                    
                    map_matrix[index] = (float)(derivative);
                    map_matrix[ (j * dim) + i ] = (float)(derivative);
                }
            }
        }

        gradients[hyp_index] = cblas_dsdot((dim*dim), chol_Q, 1, &map_matrix[0], 1);
        gradients[hyp_index] = gradients[hyp_index]/2.0;

    }
    delete[] rsq;
}

void c_kernel_LMC_SM::compute_cross_gram_matrix(
                                            const vector<int> &meta, 
                                            const vector<int> &meta2, 
                                            const vector<float> &x, 
                                            const vector<float> &x2, 
                                            float *&cross_gram_matrix
                                            ){
    int dim1, dim2;
    int q, i, j;
    dim1 = int(x.size());
    dim2 = int(x2.size());
    
    int     Q = kernel_param[0];
    int     D = kernel_param[1];
    int     R = kernel_param[2];

    float   temp_k_value;
    double  mu, v;

    float       *rsq;
    rsq         = new float[dim1*dim2];
    compute_squared_dist(x, x2, rsq);
    
    for(q = 0; q < Q; q++){
        mu = kernel_hyp[Q*D*R + q];
        v  = kernel_hyp[Q*(D*R + 1) + q];           
        for(i = 0; i < dim1; i++){
            #pragma omp parallel for private(j) firstprivate(dim1, dim2, mu, v, q, i)
            for(j = 0; j < dim2; j++){
                int index = (i * dim2) + j;
                float curr_b_coef = coregional_matrix[q][meta[i]*D + meta2[j]];
                double curr_element = curr_b_coef*compute_k(rsq[index], mu, v);
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

float c_kernel_LMC_SM::compute_k(const float &rsq, const double &mu, const double &v){
    double value;
    value = cos(2.0*PI*sqrt(rsq)*mu)*exp(-2.0*pow(PI*v, 2.0)*rsq);
    return (float) value;
}
float c_kernel_LMC_SM::compute_km(const float &rsq, const double &mu, const double &v){
    double value;
    double dmu = 2.0*PI*sqrt(rsq)*mu;
    value = (-1.0)*dmu*sin(dmu)*exp(-2.0*pow(PI*v, 2.0)*rsq);
    return (float) value;
}
float c_kernel_LMC_SM::compute_kv(const float &rsq, const double &mu, const double &v){
    double value;
    double d2piv = pow(PI*v, 2.0)*rsq;
    value = cos(2.0*PI*sqrt(rsq)*mu)*exp(-2.0*d2piv);
    value *= -4.0*(d2piv);
    return (float) value;
}
