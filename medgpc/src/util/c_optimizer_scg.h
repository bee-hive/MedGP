/*
-------------------------------------------------------------------------
This is the header file for the class scaled conjugate gradient descent.
-------------------------------------------------------------------------
*/
#ifndef C_OPTIMIZER_SCG_H
#define C_OPTIMIZER_SCG_H
#include <iostream>
#include <vector>
#include <string>

#include "util/c_optimizer.h"
#include "util/c_objective.h"
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

using namespace std;

class c_optimizer_scg:public c_optimizer{

    public:
        c_optimizer_scg(){
            optimizer_name = "c_optimizer_scg";
        };
        void optimize(
                        const int &max_iteration,
                        const vector<double> &init_parameter,
                        c_objective *objfunc,
                        const bool &display,
                        double &opt_loss, vector<double> &opt_parameter,
                        c_kernel        *&input_kernel, 
                        c_meanfunc      *&input_meanfunc, 
                        c_likelihood    *&input_likfunc, 
                        c_inference     *&input_inffunc,
                        c_prior         *&input_prior
                        );
};
#endif