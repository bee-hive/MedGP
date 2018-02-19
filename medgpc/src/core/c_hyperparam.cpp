/*
-------------------------------------------------------------------------
This is the main file for hyperparameter class.
Need to include header file: c_hyperparam.h
-------------------------------------------------------------------------
*/
#include <vector>
#include "core/c_hyperparam.h"
using namespace std;


c_hyperparam::c_hyperparam(){}

c_hyperparam::c_hyperparam(const c_hyperparam &hyp){
    set_hyp_cov(hyp.hyp_cov);
    set_hyp_mean(hyp.hyp_mean);
    set_hyp_lik(hyp.hyp_lik);
}

c_hyperparam::c_hyperparam(
                            const vector<double> &hyp_all, 
                            const int &num_cov, 
                            const int &num_mean, 
                            const int &num_lik
                            ){
    set_hyp_all(hyp_all, num_cov, num_mean, num_lik);
}

c_hyperparam::c_hyperparam(
                            const vector<double> &input_hyp_cov, 
                            const vector<double> &input_hyp_mean, 
                            const vector<double> &input_hyp_lik
                            ){
    set_hyp_cov(input_hyp_cov);
    set_hyp_mean(input_hyp_mean);
    set_hyp_lik(input_hyp_lik);
}

int c_hyperparam::get_num_hyp_cov(){
    return int(hyp_cov.size());
}

int c_hyperparam::get_num_hyp_mean(){
    return int(hyp_mean.size());
}

int c_hyperparam::get_num_hyp_lik(){
    return int(hyp_lik.size());
}

int c_hyperparam::get_num_hyp_all(){
    return int(get_num_hyp_cov()+get_num_hyp_mean()+get_num_hyp_lik());
}

vector<double> c_hyperparam::get_hyp_cov(){
    vector<double> hyp_cov_copy(hyp_cov);
    return hyp_cov_copy;
}
vector<double> c_hyperparam::get_hyp_mean(){
    vector<double> hyp_mean_copy(hyp_mean);
    return hyp_mean_copy;
}
vector<double> c_hyperparam::get_hyp_lik(){
    vector<double> hyp_lik_copy(hyp_lik);
    return hyp_lik_copy;
}

vector<double> c_hyperparam::get_hyp_all(){
    vector<double> total_hyp;
    int i;
    
    for(i = 0; i < get_num_hyp_lik(); i++){
        total_hyp.push_back(hyp_lik[i]);
    }

    for(i = 0; i < get_num_hyp_cov(); i++){
        total_hyp.push_back(hyp_cov[i]);
    }

    for(i = 0; i < get_num_hyp_mean(); i++){
        total_hyp.push_back(hyp_mean[i]);
    }

    return total_hyp;
}

void c_hyperparam::set_hyp_cov(const vector<double> &input_hyp_cov){
    hyp_cov = input_hyp_cov;
}

void c_hyperparam::set_hyp_mean(const vector<double> &input_hyp_mean){
    hyp_mean = input_hyp_mean;
}

void c_hyperparam::set_hyp_lik(const vector<double> &input_hyp_lik){
    hyp_lik = input_hyp_lik;
}

void c_hyperparam::set_hyp_all(
                                const vector<double> &input_hyp, 
                                const int &num_cov, 
                                const int &num_mean, 
                                const int &num_lik
                                ){
    hyp_cov.clear();
    hyp_mean.clear();
    hyp_lik.clear();

    int i;
    for(int i = 0; i < num_lik; i ++){
        hyp_lik.push_back(input_hyp[i]);
    }

    for(int i = 0; i < num_cov; i ++){
        hyp_cov.push_back(input_hyp[i+(num_lik)]);
    }

    for(int i = 0; i < num_mean; i ++){
        hyp_mean.push_back(input_hyp[i+(num_lik)+(num_cov)]);
    }

}


