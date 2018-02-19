/*
-------------------------------------------------------------------------
This is the function file for top mean function class.
All other mean functions can inherit these functions.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <mkl.h>
// #include <omp.h>

#include "mean/c_meanfunc.h"
#include "util/global_settings.h"

using namespace std;

c_meanfunc::c_meanfunc(){
    meanfunc_name = "c_meanfunc";
    meanfunc_hyp_num = 0;
}
c_meanfunc::c_meanfunc(vector<int> input_param){
    meanfunc_name = "c_meanfunc";
    meanfunc_hyp_num = 0;
    set_meanfunc_param(input_param);
}
c_meanfunc::c_meanfunc(vector<int> input_param, vector<double> input_hyp){
    meanfunc_name = "c_meanfunc";
    meanfunc_hyp_num = 0;
    set_meanfunc_param(input_param);
    set_meanfunc_hyp(input_hyp);
}

void c_meanfunc::print_meanfunc(){
    cout << "current mean function object: " << meanfunc_name << endl;  
}
void c_meanfunc::set_meanfunc_hyp(vector<double> input_hyp){
    meanfunc_hyp = input_hyp;
    for(int i = 0; i < int(input_hyp.size()); i++){
        meanfunc_hyp[i] = meanfunc_hyp[i];
    }
}
void c_meanfunc::set_meanfunc_param(vector<int> input_param){
    meanfunc_param = input_param;
}
        
vector<int> c_meanfunc::get_meanfunc_param(){
    vector<int> meanfunc_param_copy(meanfunc_param);
    return meanfunc_param_copy;
}
vector<double> c_meanfunc::get_meanfunc_hyp(){
    vector<double> meanfunc_hyp_copy(meanfunc_hyp);
    return meanfunc_hyp_copy;
}
int c_meanfunc::get_meanfunc_hyp_num(){
    return meanfunc_hyp_num;
}