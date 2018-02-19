/*
-------------------------------------------------------------------------
This is the header file for the class scaled conjugate gradient descent.
-------------------------------------------------------------------------
*/
#ifndef C_OPTIMIZER_varEM_H
#define C_OPTIMIZER_varEM_H
#include <iostream>
#include <vector>
#include <string>

#include "util/c_optimizer.h"
#include "util/c_objective.h"
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

#define DEFAULT_SCG_MAX_ITER 80

using namespace std;

class c_optimizer_varEM:public c_optimizer{

    public:
        c_optimizer_varEM(){
            optimizer_name = "c_optimizer_varEM";
            sub_opt_iter = DEFAULT_SCG_MAX_ITER;
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
        double update_tau(
                        const float &gamma,
                        const float &d, 
                        const float &eta,
                        const double &phi
                        );
        double update_phi(
                        const int &D,
                        const float &beta, 
                        const float &gamma,
                        const double &delta_sum,
                        const double &tau
                        );
        double update_delta(
                        const float &alpha,
                        const float &beta, 
                        const double &psi,
                        const double &phi
                        );
        double update_psi(
                        const float &alpha,
                        const double &a, 
                        const double &delta
                        );
        void set_sub_opt_iter(const int &opt_iter);

    private:
        int     sub_opt_iter;
};
#endif