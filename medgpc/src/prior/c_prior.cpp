/*
-------------------------------------------------------------------------
This is the function file for top prior class.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <math.h>
#include "prior/c_prior.h"
#include "util/global_settings.h"

using namespace std;

c_prior::c_prior(){
    hyp_cov_num = 0;
    hyp_mean_num = 0;
    hyp_lik_num = 0;
}

c_prior::c_prior(
                int num_cov, 
                int num_mean, 
                int num_lik
                ){
    initialize_param(num_cov, num_mean, num_lik);
    hyp_cov_num     = num_cov;
    hyp_mean_num    = num_mean;
    hyp_lik_num     = num_lik;
}
// copy constructor
c_prior::c_prior(const c_prior &prior){
    flag_cov    = prior.flag_cov;
    flag_mean   = prior.flag_mean;
    flag_lik    = prior.flag_lik;

    exp_cov     = prior.exp_cov;
    exp_mean    = prior.exp_mean;
    exp_lik     = prior.exp_lik;

    fix_param_cov   = prior.fix_param_cov;
    fix_param_mean  = prior.fix_param_mean;
    fix_param_lik   = prior.fix_param_lik;

    type_cov    = prior.type_cov;
    type_mean   = prior.type_mean;
    type_lik    = prior.type_lik;

    cov_varEM       = prior.cov_varEM;
    cov_varEM_fix   = prior.cov_varEM_fix;
    hyp_cov_num     = prior.hyp_cov_num;
    hyp_mean_num    = prior.hyp_mean_num;
    hyp_lik_num     = prior.hyp_lik_num;
}

void c_prior::initialize_param( 
                                int num_cov, 
                                int num_mean, 
                                int num_lik
                                ){
    fix_param_cov.clear();
    fix_param_mean.clear();
    fix_param_lik.clear();
    flag_cov.clear();
    exp_cov.clear();
    type_cov.clear();
    flag_mean.clear();
    exp_mean.clear();
    type_mean.clear();
    flag_lik.clear();
    exp_lik.clear();
    type_lik.clear();
    cov_varEM.clear();
    cov_varEM_fix.clear();

    for(int i = 0; i < num_cov; i++){
        vector<float> a;
        fix_param_cov.push_back(a);
        flag_cov.push_back(false);
        exp_cov.push_back(false);
        type_cov.push_back(-1);
    }
    for(int i = 0; i < num_mean; i++){
        vector<float> a;
        fix_param_mean.push_back(a);
        flag_mean.push_back(false);
        exp_mean.push_back(false);
        type_mean.push_back(-1);
    }
    for(int i = 0; i < num_lik; i++){
        vector<float> a;
        fix_param_lik.push_back(a);
        flag_lik.push_back(false);
        exp_lik.push_back(false);
        type_lik.push_back(-1);
    }

    test_mode_parameter.clear();
    for(int i = 0; i < (num_lik + num_cov + num_mean); i++){
        test_mode_parameter.push_back(0.0);
    }
    cout << "init. length of mode param = " << test_mode_parameter.size() << endl;
}

void c_prior::init_cov_varEM(int num_cov_varEM, double init_val){
    vector<double> init_vec(num_cov_varEM, init_val);
    cov_varEM = init_vec;
}
void c_prior::init_cov_varEM_fix(int num_cov_varEM_fix, double init_val){
    vector<double> init_vec(num_cov_varEM_fix, init_val);
    cov_varEM_fix = init_vec;
}

void c_prior::init_test_prior(const int kernel_index, 
                              const vector<int> &test_kernel_param,
                              const vector<double> &test_mode_param
                              ){
    if(kernel_index != 7){
        cout << "Warning: testing prior is only set for LMCSM kernel now;"
             << "prior will not be effective" << endl;
    }
    else{
        cout << "Info: setup prior to fix zero A elements" << endl;
        int Q = test_kernel_param[0];
        int D = test_kernel_param[1];
        int R = test_kernel_param[2];

        // covariance
        for(int i = D; i < (D+Q*D*R); i++){
            if(test_mode_param[i] == 0.0){
                flag_cov[i-D] = true;
                type_cov[i-D] = 0; // clamped prior
            }
        }
    }
}
bool c_prior::get_one_prior_flag(const int &index){
    if(index < hyp_lik_num){
        return flag_lik[index];
    }
    else if(index < (hyp_lik_num + hyp_cov_num)){
        return flag_cov[index - hyp_lik_num];
    }
    else{
        return flag_mean[index - hyp_lik_num - hyp_cov_num];
    }
}

int c_prior::get_one_prior_type(const int &index){
    if(index < hyp_lik_num){
        return type_lik[index];
    }
    else if(index < (hyp_lik_num + hyp_cov_num)){
        return type_cov[index - hyp_lik_num];
    }
    else{
        return type_mean[index - hyp_lik_num - hyp_cov_num];
    }
}

vector<double> c_prior::prior_lik_kde(
                            const double &x, 
                            const vector<float> &param
                            ){
    vector<double> lik_vec;
    double lp, dlp;
    float bw = param[0];
    int num_sample = int(param.size()) - 1;
    
    lp = 0.0;
    dlp = 0.0;
    for(int i = 0; i < num_sample; i++){
        double ds, dsd;
        ds = exp(-0.5*pow((x - param[i+1])/bw, 2.0))/sqrt(2*PI);
        dsd = (x - param[i+1])*ds;
        lp += ds;
        dlp += dsd;
    }
    lp = lp / (float(num_sample)*bw);
    dlp = -1.0*dlp/(float(num_sample)*pow(bw, 3.0f));
    dlp = dlp / lp;

    lp = log(lp);

    lik_vec.push_back(lp);
    lik_vec.push_back(dlp);

    return lik_vec;

}

void c_prior::setup_param(
                            const int kernel_index, 
                            const vector<int> &kernel_param, 
                            const int &mode,
                            const vector<float> &prior_param
                            ){
    if(kernel_index != 7){
        cout << "Warning: prior mode is only available for LMCSM kernel now; "
             << "prior will not be effective" << endl;
    }
    else{
        switch(mode){
            case 0:
                cout << "mode " << mode << ": no regularization" << endl;
                break;
            case 2:
                cout << "mode " << mode << ": setup hierarchical gamma prior" << endl;
                setup_hier_gamma_prior(kernel_param, prior_param);
                break;
            default:
                cout << "undefined setup mode " << mode << "; no changes" << endl;
                break;
        }
    }
}

void c_prior::setup_hier_gamma_prior(
                                        const vector<int> &kernel_param, 
                                        const vector<float> &prior_param
                                        ){
    int i, Q, D, R;
    Q = kernel_param[0];
    D = kernel_param[1];
    R = kernel_param[2];

    // cout << "setup hierarchical gamma prior Q/D/R = " << Q << "/" << D << "/" << R << endl;

    // initialize variational EM parameters
    init_cov_varEM(2*Q*(D*R+R), 1.0);
    init_cov_varEM_fix(5, 0.5);
    if(prior_param.size() > 0){
        set_cov_varEM_fix_one(prior_param[0], 4);   
    }
    else{
        set_cov_varEM_fix_one(50.0, 4);
    }

    // setup individual priors
    for(i = 0; i < hyp_cov_num; i++){
        if(i < Q*D*R){ // a
            flag_cov[i] = true;
            exp_cov[i] = false;
            fix_param_cov[i].clear();
            fix_param_cov[i].push_back(0.0);
            fix_param_cov[i].push_back(1.0);
            type_cov[i] = 1; // normal
        }
        else if(i < Q*(D*R+1)){ // mu
            flag_cov[i] = false;
            exp_cov[i] = true;
        }   
        else if(i < Q*(D*R+2)){ // v
            flag_cov[i] = false;
            exp_cov[i] = true;
        }
        else if(i < Q*(D*R+2+D)){ // lambda
            flag_cov[i] = true;
            exp_cov[i] = true;
            fix_param_cov[i].clear();
            fix_param_cov[i].push_back(0.0);
            if(prior_param.size() > 1){
                fix_param_cov[i].push_back(prior_param[1]);
            }
            else{
                fix_param_cov[i].push_back(0.5);
            }           
            type_cov[i] = 2; // laplace
        }
        else{ // const
            flag_cov[i] = false;
            exp_cov[i] = true;
        }
    }
}

void c_prior::set_cov_varEM_all(const vector<double> &input_varEM){
    cov_varEM = input_varEM;
}
vector<double> c_prior::get_cov_varEM_all(){
    vector<double> cov_varEM_copy(cov_varEM);
    return cov_varEM_copy;
}
void c_prior::set_cov_varEM_one(double value, const int &index){
    cov_varEM[index] = value;
}
double c_prior::get_cov_varEM_one(const int &index){
    double a = cov_varEM[index];
    return a;
}

void c_prior::set_cov_varEM_fix_all(const vector<double> &input_varEM_fix){
    cov_varEM_fix = input_varEM_fix;
}
vector<double> c_prior::get_cov_varEM_fix_all(){
    vector<double> cov_varEM_fix_copy(cov_varEM_fix);
    return cov_varEM_fix_copy;
}
void c_prior::set_cov_varEM_fix_one(double value, const int &index){
    cov_varEM_fix[index] = value;
}
double c_prior::get_cov_varEM_fix_one(const int &index){
    double a = cov_varEM_fix[index];
    return a;
}

vector<double> c_prior::get_one_lik_cov(
                                        const double &x, 
                                        const int &index
                                        ){
    double input_x = x;
    vector<double> lik;
    switch(type_cov[index]){
        case 1:
            lik = prior_lik_normal(input_x, fix_param_cov[index]);
            break;
        case 2:
            lik = prior_lik_laplace(input_x, fix_param_cov[index]);
            break;
        case 3:
            lik = prior_lik_kde(input_x, fix_param_cov[index]);
            break;
        default:
            lik.push_back(0.0);
            lik.push_back(0.0);
            break;
    }
    return lik;
}

vector<double> c_prior::get_one_lik_lik(
                                        const double &x, 
                                        const int &index
                                        ){
    double input_x = x;
    vector<double> lik;
    switch(type_lik[index]){
        case 1:
            lik = prior_lik_normal(input_x, fix_param_lik[index]);
            break;
        case 2:
            lik = prior_lik_laplace(input_x, fix_param_lik[index]);
            break;
        case 3:
            lik = prior_lik_kde(input_x, fix_param_lik[index]);
            break;
        default:
            lik.push_back(0.0);
            lik.push_back(0.0);
            break;
    }
    return lik;
}

vector<double> c_prior::get_one_lik_mean(
                                        const double &x, 
                                        const int &index
                                        ){
    double input_x = x;
    vector<double> lik;
    switch(type_mean[index]){
        case 1:
            lik = prior_lik_normal(input_x, fix_param_mean[index]);
            break;
        case 2:
            lik = prior_lik_laplace(input_x, fix_param_mean[index]);
            break;
        case 3:
            lik = prior_lik_kde(input_x, fix_param_mean[index]);
            break;
        default:
            lik.push_back(0.0);
            lik.push_back(0.0);
            break;
    }
    return lik;
}

vector<double> c_prior::prior_lik_normal(
                                        const double &x, 
                                        const vector<float> &param
                                        ){
    vector<double> lik_vec;
    double lp, dlp;
    lp = -1.0*(x - param[0])*(x - param[0])/(2.0*param[1]);
    lp = lp - log(2*PI*param[1])/2.0;
    lik_vec.push_back(lp);

    dlp = -1.0*(x - param[0])/param[1];
    lik_vec.push_back(dlp);
    return lik_vec;
}

vector<double> c_prior::prior_lik_laplace(
                                         const double &x, 
                                         const vector<float> &param
                                         ){
    vector<double> lik_vec;
    double lp, dlp;
    lp = (-1.0*fabs(x - param[0])/param[1]) - log(2*param[1]);
    lik_vec.push_back(lp);
    
    if(x == param[0]){
        dlp = 0.0;
    }
    else{
        if(x > param[0]){
            dlp = 1.0;
        }
        else{
            dlp = -1.0;
        }
        dlp = -1.0*dlp/param[1];
    }
    lik_vec.push_back(dlp);
    return lik_vec;
}

void c_prior::print_status(){
    cout << "print current prior status:" << endl;
    cout << "setup cov num = " << hyp_cov_num << endl;
    cout << "setup mean num = " << hyp_mean_num << endl;
    cout << "setup lik num = " << hyp_lik_num << endl;

    cout << endl;
    cout << "checking prior parameters length..." << endl;
    cout << "length of active flag (cov/mean/lik) = " 
            << flag_cov.size() << "/"
            << flag_mean.size() << "/"
            << flag_lik.size() << endl;
    cout << "length of exp. transform flag (cov/mean/lik) = " 
            << exp_cov.size() << "/"
            << exp_mean.size() << "/"
            << exp_lik.size() << endl;
    cout << "length of type flag (cov/mean/lik) = " 
            << type_cov.size() << "/"
            << type_mean.size() << "/"
            << type_lik.size() << endl;
    cout << "length of fix parameters (cov/mean/lik) = " 
            << fix_param_cov.size() << "/"
            << fix_param_mean.size() << "/"
            << fix_param_lik.size() << endl;

    cout << endl;
    cout << "checking prior type and status..." << endl;
    for(int i = 0; i < hyp_cov_num; i++){
        cout << "prior for cov[" << i << "]: ";
        print_one_prior(flag_cov[i], type_cov[i], fix_param_cov[i]);
    }
    for(int i = 0; i < hyp_mean_num; i++){
        cout << "prior for mean[" << i << "]: ";
        print_one_prior(flag_mean[i], type_mean[i], fix_param_mean[i]);
    }
    for(int i = 0; i < hyp_lik_num; i++){
        cout << "prior for lik[" << i << "]: ";
        print_one_prior(flag_lik[i], type_lik[i], fix_param_lik[i]);
    }

    cout << endl;
    cout << "checking variational EM (varEM) parameters..." << endl;
    cout << "# covariance varEM fix parameters = " << cov_varEM_fix.size() << endl;
    for(int i = 0; i < int(cov_varEM_fix.size()); i++){
        cout << "cov_varEM_fix[" << i << "] = " << cov_varEM_fix[i] << endl; 
    }
    cout << "# covariance varEM parameters = " << cov_varEM.size() << endl;
    for(int i = 0; i < int(cov_varEM.size()); i++){
        cout << "cov_varEM[" << i << "] = " << cov_varEM[i] << endl; 
    }
    cout << "Finish printing priors" << endl;
    cout << endl;

}

void c_prior::print_one_prior(
                                const bool &flag, 
                                const int &type, 
                                const vector<float> &param){
    if(flag){
        cout << "active, ";
    }
    else{
        cout << "not active, ";
    }
    
    switch(type){
        case -1:
            cout << "NaN";
            break;
        case 0:
            cout << "clamped";
            break;
        case 1:
            cout << "normal( ";
            for(int i = 0; i < int(param.size()); i++){
                cout << param[i];
                if(i != (int(param.size())-1)) 
                    cout << ", ";
            }
            cout << ")";
            break;
        case 2:
            cout << "laplace( ";
            for(int i = 0; i < int(param.size()); i++){
                cout << param[i];
                if(i != (int(param.size())-1)) 
                    cout << ", ";
            }
            cout << ")";
            break;
        default:
            cout << "not defined!";
            break;
    }
    cout << endl;
}


