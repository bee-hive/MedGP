/*
-------------------------------------------------------------------------
This is the function file for the class objective of individuals.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <string>

#include "util/c_objective_one.h"
#include "util/global_settings.h"
#include "core/c_hyperparam.h"
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

using namespace std;

c_objective_one::c_objective_one(){
    objective_name = "c_objective_one";
    dist_thread_num = 1;
}

c_objective_one::c_objective_one(
                const int &kernel_idx,
                const vector<int> &kernel_param,
                const vector<int> &meta, 
                const vector<float> &x, 
                const vector<float> &y
                ){
    objective_name = "c_objective_one";
    obj_meta = meta;
    obj_x = x;
    obj_y = y;
    obj_kernel_idx      = kernel_idx;
    obj_kernel_param    = kernel_param;
}
void c_objective_one::set_dist_thread_num(int input_thread_num){
    dist_thread_num = 1;
}
bool c_objective_one::compute_objective(
                        const bool &flag_grad, 
                        const vector<double> &input_parameter, 
                        double          &objective_value, 
                        vector<double>  &gradients,
                        c_kernel        *&input_kernel, 
                        c_meanfunc      *&input_meanfunc, 
                        c_likelihood    *&input_likfunc, 
                        c_inference     *&input_inffunc,
                        c_prior         *&input_prior
                        ){
    if(int(obj_x.size()) > 2){
        int iter;

        c_hyperparam    hyp(
                            input_parameter, 
                            input_kernel -> get_kernel_hyp_num(), 
                            input_meanfunc -> get_meanfunc_hyp_num(), 
                            input_likfunc -> get_likfunc_hyp_num()
                            );
        input_kernel -> set_kernel_hyp(hyp.get_hyp_cov());
        input_meanfunc -> set_meanfunc_hyp(hyp.get_hyp_mean());
        input_likfunc -> set_likfunc_hyp(hyp.get_hyp_lik());

        GP_Regression   curr_gpr_model(1, input_kernel, input_meanfunc, input_likfunc, input_inffunc, input_prior);
        
        curr_gpr_model.train(flag_grad, obj_meta, obj_x, obj_y);

        if(!curr_gpr_model.get_flag_trained()){
            return false;
        }
        else{
            objective_value = curr_gpr_model.get_neg_log_mlikelihood();
            if(flag_grad){
                gradients = curr_gpr_model.get_dneg_log_mlikelihood();          
            }
            return true;
        }
    }
    else{
        return false;
    }
}