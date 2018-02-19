/*
-------------------------------------------------------------------------
This is the header file for linear model of coregional + spectral mixture kernel.
-------------------------------------------------------------------------
*/
#ifndef C_KERNEL_LMC_SM_H
#define C_KERNEL_LMC_SM_H
#include <vector>
#include "kernel/c_kernel.h"

using namespace std;

class c_kernel_LMC_SM:public c_kernel{

    public:
        c_kernel_LMC_SM();
        c_kernel_LMC_SM(const vector<int> &input_param);
        c_kernel_LMC_SM(const vector<int> &input_param, const vector<double> &input_hyp);

        void compute_self_diag_matrix(
                                        const vector<int> &meta, 
                                        const vector<float> &x, 
                                        float *&diag_gram_matrix
                                        );

        void compute_self_gram_matrix(
                                        const vector<int> &meta, 
                                        const vector<float> &x,
                                        float *&self_gram_matrix
                                        );

        void compute_self_gradients(
                                    const vector<int> &meta, 
                                    const vector<float> &x,
                                    const float *chol_Q, 
                                    vector<double> &gradients
                                    );
        
        void compute_cross_gram_matrix(
                                        const vector<int> &meta, 
                                        const vector<int> &meta2, 
                                        const vector<float> &x, 
                                        const vector<float> &x2, 
                                        float *&cross_gram_matrix
                                        );
        // compute specific derivatives
        void    compute_coregional_matrix();
        void    reset_coregional_matrix(vector< vector<double> > em_B_array);

        float   compute_k(const float &rsq, const double &mu, const double &v);
        float   compute_km(const float &rsq, const double &mu, const double &v);
        float   compute_kv(const float &rsq, const double &mu, const double &v);

        void    set_kernel_param(const vector<int> &input_param);
        void    set_kernel_hyp(const vector<double> &input_hyp);

    private:
        vector< vector<double> >    coregional_matrix;

        
};
#endif