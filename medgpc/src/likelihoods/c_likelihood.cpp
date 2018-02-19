/*
-------------------------------------------------------------------------
This is the function file for top likelihood function class.
All other likelihood functions can inherit these functions.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <mkl.h>

#include "likelihoods/c_likelihood.h"

using namespace std;

c_likelihood::c_likelihood(){
    likfunc_name = "c_likelihood";
    likfunc_hyp_num = 0;
}

c_likelihood::c_likelihood(vector<int> input_param){
    likfunc_name = "c_likelihood";
    set_likfunc_param(input_param);
}

c_likelihood::c_likelihood(vector<int> input_param, vector<double> input_hyp){
    likfunc_name = "c_likelihood";
    likfunc_hyp_num = 0;

    set_likfunc_param(input_param);
    set_likfunc_hyp(input_hyp);
}

void c_likelihood::print_likfunc(){
    cout << "current likelihood function object: " << likfunc_name << endl; 
}
void c_likelihood::set_likfunc_hyp(vector<double> input_hyp){
    likfunc_hyp = input_hyp;
    for(int i = 0; i < int(input_hyp.size()); i++){
        likfunc_hyp[i] = exp(likfunc_hyp[i]);
    }
}
void c_likelihood::set_likfunc_param(vector<int> input_param){
    likfunc_param = input_param;
}
        
vector<int> c_likelihood::get_likfunc_param(){
    vector<int> likfunc_param_copy(likfunc_param);
    return likfunc_param_copy;
}
vector<double> c_likelihood::get_likfunc_hyp(){
    vector<double> likfunc_hyp_copy(likfunc_hyp);
    return likfunc_hyp_copy;
}
int c_likelihood::get_likfunc_hyp_num(){
    return likfunc_hyp_num;
}