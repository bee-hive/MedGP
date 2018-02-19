/*
-------------------------------------------------------------------------
This is the header file for the class inference.
Gaussian(MO) likelihood can be put into inference class directly.
-------------------------------------------------------------------------
*/
#ifndef C_INFERENCE_H
#define C_INFERENCE_H
#include <iostream>
#include <vector>
#include <string>

#include "kernel/c_kernel.h"
#include "mean/c_meanfunc.h"
#include "likelihoods/c_likelihood.h"
#include "prior/c_prior.h"

using namespace std;

class c_inference{

    public:
        c_inference(){
            inffunc_name = "c_inference";
            inf_thread_num = 1;
        };
        c_inference(const int &thread_num){
            inffunc_name = "c_inference";
            inf_thread_num = thread_num;
        };
        void print_inffunc(){
            cout << "current inference object: " << inffunc_name 
                 << "; number of threads: " << inf_thread_num << endl;
        };
        int get_thread_num(){
            return inf_thread_num;
        };
        virtual bool compute_nlml(
                                const bool &flag_grad, 
                                const vector<int> &meta, 
                                const vector<float> &x, 
                                const vector<float> &y, 
                                c_kernel *kernel, 
                                c_meanfunc *meanfunc, 
                                c_likelihood *likfunc, 
                                c_prior *prior,
                                float *&chol_alpha, 
                                float *&chol_factor_inv, 
                                float &beta,
                                double &nlml, 
                                vector<double> &dnlml
                                ){ cout << "compute_nlml in c_inference" << endl; };

    protected:
        string  inffunc_name;
        int     inf_thread_num;
};
#endif