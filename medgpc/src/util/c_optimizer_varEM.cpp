/*
-------------------------------------------------------------------------
This is the function file for the class scaled conjugate gradient descent.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>

#include "util/c_optimizer_varEM.h"
#include "util/c_optimizer_scg.h"
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

using namespace std;

void c_optimizer_varEM::optimize(
                                const int &max_iteration,
                                const vector<double> &init_parameter,
                                c_objective *objfunc,
                                const bool &display,
                                double &opt_loss, 
                                vector<double> &opt_parameter,
                                c_kernel        *&input_kernel, 
                                c_meanfunc      *&input_meanfunc, 
                                c_likelihood    *&input_likfunc, 
                                c_inference     *&input_inffunc,
                                c_prior         *&input_prior
                                ){
    int iter, Q, D, R;
    int index, q, d, r, offset;
    float alpha, beta, gamma, dd, eta;
    double psi, delta, phi, tau;
    vector<int> kernel_param;

    // get LMCSM kernel settings
    kernel_param = (input_kernel -> get_kernel_param());
    if(int(kernel_param.size()) != 3){
        cout << "ERROR: varEM is only usable for LMCSM kernel!" << endl;
        exit(1);
    }
    Q = kernel_param[0];
    D = kernel_param[1];
    R = kernel_param[2];

    // initialize functions and parameters
    c_optimizer_scg     curr_optfunc;
    opt_parameter = init_parameter;
    double best_loss;

    for(iter = 0; iter < abs(max_iteration); iter++){
        // update through scg
        vector<double> scg_init_parameter(opt_parameter);
        int curr_sub_opt_iter;
        if(iter < 5){
            curr_sub_opt_iter = 100;
        }
        else{
            curr_sub_opt_iter = sub_opt_iter;
        }
        // curr_sub_opt_iter = sub_opt_iter;
        curr_optfunc.optimize(
                                (-1)*curr_sub_opt_iter,
                                scg_init_parameter,
                                objfunc,
                                false,
                                opt_loss, 
                                opt_parameter,
                                input_kernel, 
                                input_meanfunc, 
                                input_likfunc, 
                                input_inffunc,
                                input_prior
                                );
        if(display){
            cout << "iteration " << iter << " for variational EM:" 
                 << " loss = " << opt_loss << endl;
        }
        
        if(iter > 0){
            double change_ratio = (opt_loss-best_loss)/best_loss;
            if( abs(change_ratio) < 0.005 ){
                cout << "change of loss " << change_ratio << " meets early stop criterion" << endl;
                break;
            }
        }
        best_loss = opt_loss;

        alpha = (input_prior -> get_cov_varEM_fix_one(0));
        beta = (input_prior -> get_cov_varEM_fix_one(1));
        gamma = (input_prior -> get_cov_varEM_fix_one(2));
        dd = (input_prior -> get_cov_varEM_fix_one(3));
        eta = (input_prior -> get_cov_varEM_fix_one(4));

        // update tau
        for(q = 0; q < Q; q++){
            for(r = 0; r < R; r++){
                index = Q*(2*D*R+R) + q*R + r;
                phi = (input_prior -> get_cov_varEM_one((index - Q*R)));
                input_prior -> set_cov_varEM_one(update_tau(gamma, dd, eta, phi), index);
            }
        }

        // update phi
        for(q = 0; q < Q; q++){
            for(r = 0; r < R; r++){
                index = Q*(2*D*R) + q*R + r;
                double delta_sum = 0.0;
                for(d = 0; d < D; d++){
                    delta_sum += (input_prior -> get_cov_varEM_one((Q*D*R + q*D*R + d*R + r)));
                }
                tau = (input_prior -> get_cov_varEM_one((index + Q*R)));
                input_prior -> set_cov_varEM_one(update_phi(D, beta, gamma, delta_sum, tau), index);

            }
        }

        // update delta
        for(q = 0; q < Q; q++){
            for(d = 0; d < D; d++){
                for(r = 0; r < R; r++){
                    index = Q*D*R + q*D*R + d*R + r;
                    psi = (input_prior -> get_cov_varEM_one((index - Q*D*R)));
                    phi = (input_prior -> get_cov_varEM_one((2*Q*D*R + q*R + r)));
                    input_prior -> set_cov_varEM_one(update_delta(alpha, beta, psi, phi), index);
                }
            }
        }

        // update psi
        offset = (input_likfunc -> get_likfunc_hyp_num());
        for(q = 0; q < Q; q++){
            for(d = 0; d < D; d++){
                for(r = 0; r < R; r++){
                    // get index and a
                    index = q*D*R + d*R + r;
                    double a = opt_parameter[offset + index];
                    delta = (input_prior -> get_cov_varEM_one((index + Q*D*R)));
                    input_prior -> set_cov_varEM_one(update_psi(alpha, a, delta), index);

                    // if psi = 0, assign a = 0, and deactivate clamp the prior of a
                    if((input_prior -> get_cov_varEM_one(index)) == 0.0){
                        (input_prior -> type_cov)[index] = 0;
                        opt_parameter[offset + index] = 0.0;
                    }

                    // update linked prior for a
                    (input_prior -> fix_param_cov)[index][0] = 0;
                    (input_prior -> fix_param_cov)[index][1] = (input_prior -> get_cov_varEM_one(index));
                }
            }
        }
    }
}

double c_optimizer_varEM::update_tau(
                                    const float &gamma,
                                    const float &d, 
                                    const float &eta,
                                    const double &phi
                                    ){
    double new_tau;
    new_tau = (gamma + d)/(phi + eta);
    return new_tau;
}
double c_optimizer_varEM::update_phi(
                                    const int &D,
                                    const float &beta, 
                                    const float &gamma,
                                    const double &delta_sum,
                                    const double &tau
                                    ){
    double new_phi;
    new_phi = ( ((float)(D))*beta + gamma - 1.0)/(delta_sum + tau);
    return new_phi;
}
double c_optimizer_varEM::update_delta(
                                        const float &alpha,
                                        const float &beta, 
                                        const double &psi,
                                        const double &phi
                                        ){
    double new_delta;
    new_delta = (alpha + beta)/(psi + phi);
    return new_delta;
}
double c_optimizer_varEM::update_psi(
                                    const float &alpha,
                                    const double &a, 
                                    const double &delta
                                    ){
    double new_psi, sub;
    sub = (2.0*alpha - 3.0);
    new_psi = sub + sqrt(sub*sub + 8.0*delta*a*a);
    new_psi = new_psi/(4.0*delta);
    return new_psi;
}

void c_optimizer_varEM::set_sub_opt_iter(const int &opt_iter){
    sub_opt_iter = opt_iter;
}
