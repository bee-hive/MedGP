/*
-------------------------------------------------------------------------
This is the header file for the class objective of individuals.
-------------------------------------------------------------------------
*/
#ifndef C_OBJECTIVE_ONE_H
#define C_OBJECTIVE_ONE_H
#include <iostream>
#include <vector>
#include <string>

#include "util/c_objective.h"
#include "core/gp_regression.h"

using namespace std;

class c_objective_one:public c_objective{

    public:
        c_objective_one();
        c_objective_one(
                        const int &kernel_idx,
                        const vector<int> &kernel_param,
                        const vector<int> &meta, 
                        const vector<float> &x, 
                        const vector<float> &y
                        );
        void set_dist_thread_num(int input_thread_num);
        bool compute_objective(
                                const bool &flag_grad, 
                                const vector<double> &input_parameter, 
                                double &objective_value, 
                                vector<double> &gradients,
                                c_kernel        *&input_kernel, 
                                c_meanfunc      *&input_meanfunc, 
                                c_likelihood    *&input_likfunc, 
                                c_inference     *&input_inffunc,
                                c_prior         *&input_prior
                                );
    private:
        vector<int>     obj_meta;
        vector<float>   obj_x;
        vector<float>   obj_y;
        int             obj_kernel_idx;
        vector<int>     obj_kernel_param;

};
#endif