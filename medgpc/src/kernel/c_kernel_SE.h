/*
-------------------------------------------------------------------------
This is the header file for top kernel class.
All other kernels should inherit this class.
-------------------------------------------------------------------------
*/
#ifndef C_KERNEL_SE_H
#define C_KERNEL_SE_H
#include <vector>
#include "kernel/c_kernel.h"

using namespace std;

class c_kernel_SE:public c_kernel{

    public:
        c_kernel_SE();
        c_kernel_SE(const vector<int> &input_param);
        c_kernel_SE(const vector<int> &input_param, const vector<double> &input_hyp);

        void set_kernel_param(const vector<int> &input_param);
        void set_kernel_hyp(const vector<double> &input_hyp);

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
};
#endif