/*
-------------------------------------------------------------------------
This is the function file for building experiment.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <random>
#include <iomanip>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

#include "core/gp_model_include.h"
#include "util/global_settings.h"
#include "dataio/c_experiment.h"

#define OUT_DIR_PERMIT 0774
using namespace std;
using namespace rapidjson;

void c_experiment::init_default_param(){
    srand_seed = 0;
    cv_fold_num = 10;
    scg_init_num = 100;
    scg_max_iter_num = 100;
    prior_mode = 0;
    prior_sub_opt_iter = 30;
    learn_rate = 0.00001;
    momentum = 0.9;
}

c_experiment::c_experiment(){
    init_default_param();
}

c_experiment::c_experiment(const string &input_cfg_name){
    // read in config. file
    init_default_param();

    exp_cfg_file = input_cfg_name;
    cout << "read in config. file " << exp_cfg_file << endl;

    ifstream ifs(exp_cfg_file.c_str());
    IStreamWrapper isw(ifs);
    Document d;
    d.ParseStream(isw);

    // setup data directory
    assert(d["data_dir"].IsString());
    exp_data_dir = d["data_dir"].GetString();
    exp_data_dir += "/";

    // setup experiment root directory
    assert(d["exp_top_dir"].IsString());
    exp_top_dir = d["exp_top_dir"].GetString();
    exp_top_dir += "/";

    // setup training and testing output dir
    assert(d["exp_train_dir"].IsString());
    exp_train_dir = d["exp_train_dir"].GetString();
    exp_train_dir += "/";

    assert(d["exp_test_dir"].IsString());
    exp_test_dir = d["exp_test_dir"].GetString();
    exp_test_dir += "/";

    assert(d["exp_kernel_dir"].IsString());
    exp_kernel_dir = d["exp_kernel_dir"].GetString();
    exp_kernel_dir += "/";

    // setup kernel parameters
    assert(d["kernel_index"].IsInt());
    kernel_index = d["kernel_index"].GetInt();

    kernel_param.clear();
    
    assert(d["Q"].IsInt());
    kernel_param.push_back(d["Q"].GetInt());
    
    assert(d["D"].IsInt());
    kernel_param.push_back(d["D"].GetInt());
    
    assert(d["R"].IsInt());
    kernel_param.push_back(d["R"].GetInt());
    
    for(int i = 0; i < 3; i++){
        cout << "kernel_param[" << i << "] = " << kernel_param[i] << endl;
    }

    // setup prior
    assert(d["prior_index"].IsInt());
    prior_mode = d["prior_index"].GetInt();
    if(prior_mode == 2){
        prior_hyp.clear();

        assert(d["eta"].IsFloat());
        prior_hyp.push_back(d["eta"].GetFloat());

        assert(d["beta_lam"].IsFloat());
        prior_hyp.push_back(d["beta_lam"].GetFloat()); 
    }

    // read in feature indices
    int val;

    assert(d["feature_index"].IsString());
    string fs = d["feature_index"].GetString();
    
    istringstream is(fs);
    feature_index.clear();
    for(int d = 0; d < kernel_param[1]; d++){
        is >> val;
        feature_index.push_back(val);
    }

    // optimization setup
    assert(d["random_seed"].IsInt());
    srand_seed = d["random_seed"].GetInt();

    assert(d["cv_fold_num"].IsInt());
    cv_fold_num = d["cv_fold_num"].GetInt();

    assert(d["random_init_num"].IsInt());
    scg_init_num = d["random_init_num"].GetInt();

    assert(d["top_iteration_num"].IsInt());
    scg_max_iter_num = d["top_iteration_num"].GetInt();

    assert(d["iteration_num_per_update"].IsInt());
    prior_sub_opt_iter = d["iteration_num_per_update"].GetInt();

    assert(d["online_learn_rate"].IsDouble());
    learn_rate = d["online_learn_rate"].GetDouble();

    assert(d["online_momentum"].IsDouble());
    momentum = d["online_momentum"].GetDouble();

    assert(d["exp_cfg_dir"].IsString());
    string exp_cfg_dir = d["exp_cfg_dir"].GetString();

    assert(d["hyp_bound_file"].IsString());
    string hyp_bound_file = d["hyp_bound_file"].GetString();
    exp_hyp_bound_file = exp_cfg_dir + "/" + hyp_bound_file;

    get_hyp_bounds();

    // print summary of this experiment
    print_experiment();

}
string c_experiment::get_exp_train_dir(){
    return exp_train_dir;
}
string c_experiment::get_exp_test_dir(){
    return exp_test_dir;
}

int c_experiment::get_kernel_index(){
    return kernel_index;
}
vector<int> c_experiment::get_kernel_param(){
    vector<int> kernel_param_copy(kernel_param);
    return kernel_param_copy;
}
vector<int> c_experiment::get_feature_index(){
    vector<int> feature_index_copy(feature_index);
    return feature_index_copy;
}

vector<int> c_experiment::get_test_kernel_param(int fold, const string &kernel_clust_alg){
    vector<int> test_kernel_param(kernel_param);
    // read in the new mixture number if using mixture type of kernels
    if((kernel_index == 7) or (kernel_index == 8)){
        int new_mixture_num;
        string new_mixture_file = exp_kernel_dir 
                                    + "fold" + to_string((long long) fold) + "/"
                                    + kernel_clust_alg + "_"
                                    + "mode_mixture_num.txt";
        cout << "read in new mixture number from " << new_mixture_file << endl;
        // read in kernel cluster num
        ifstream data(new_mixture_file.c_str(), ios::in);
        data >> new_mixture_num;
        data.close();
        test_kernel_param[0] = new_mixture_num;
    }
    return test_kernel_param;
}

vector<double> c_experiment::get_test_mode_param(int fold, const string &kernel_clust_alg){
    vector<double> mode_param;
    double one_param;
    string mode_file = exp_kernel_dir 
                        + "fold" + to_string((long long) fold) + "/"
                        + kernel_clust_alg + "_"
                        + "mode_param.bin";
    cout << "read in mode parameters from " << mode_file << endl;

    ifstream databin(mode_file, ios::binary);
    while(databin.read(reinterpret_cast<char*>(&one_param), sizeof(double))){
        mode_param.push_back(one_param);
    }
    databin.close();

    cout << "read in " << int(mode_param.size()) << " parameters" << endl;
    // for(int i = 0; i < int(mode_param.size()); i++){
    //  cout << mode_param[i] << '\t';
    // }
    cout << endl;
    return mode_param;
}

vector<int> c_experiment::get_lik_param(){
    vector<int> lik_param;
    if(get_lik_num() >= 1){
        lik_param.push_back(kernel_param[1]);   
    }
    return lik_param;
}
vector<float> c_experiment::get_prior_hyp(){
    vector<float> prior_hyp_copy(prior_hyp);
    return prior_hyp_copy;
}
int c_experiment::get_prior_mode(){
    return prior_mode;
}
int c_experiment::get_cv_fold_num(){
    return cv_fold_num;
}
int c_experiment::get_scg_init_num(){
    return scg_init_num;
}
int c_experiment::get_scg_max_iter_num(){
    return scg_max_iter_num;
}
int c_experiment::get_prior_sub_opt_iter(){
    return prior_sub_opt_iter;
}
double c_experiment::get_online_learn_rate(){
    return learn_rate;
}
double c_experiment::get_online_momentum(){
    return momentum;
}

void c_experiment::get_one_patient_data( 
                                        string PAN,
                                        vector<int> &meta_vec,
                                        vector<float> &time_vec, 
                                        vector<float> &value_vec
                                        ){
    float vec_len;
    float temp;
    double norm_temp;

    // load raw data
    meta_vec.clear();
    time_vec.clear();
    value_vec.clear();

    int feature_num = int(feature_index.size());

    // read subject raw data
    for(int j = 0; j < feature_num; j++){
        // read cohort mean and standard deviation
        vector<double> stat;
        double f;
        string filename = exp_data_dir + "feature" + to_string((long long) feature_index[j]) + "_stat.bin";
        ifstream databin(filename, ios::binary);
        cout << "mean/std: ";
        while(databin.read(reinterpret_cast<char*>(&f), sizeof(double))){
            cout << f << '\t';
            stat.push_back(f);
        }
        databin.close();
        cout << endl;

        // read raw data
        filename = exp_data_dir + PAN + "/";
        filename += "feature" + to_string((long long) feature_index[j]) + ".txt";

        ifstream data(filename.c_str(), ios::in);
        if(!data){
            cerr << "File " << filename << " could not be opened." << endl;
            exit(1);
        }
        cout << "reading data file " << filename << endl;
        data >> vec_len;
        for(int i = 0; i < int(vec_len); i++){
            meta_vec.push_back(j);
            
            data >> temp;
            time_vec.push_back(temp);
            
            data >> temp;
            norm_temp = ((double)temp - stat[0])/stat[1];
            value_vec.push_back((float)norm_temp);
        }
        data.close();
    }
}

int c_experiment::get_hyp_num(){
    return (get_lik_num() + get_cov_num() + get_mean_num());
}

int c_experiment::get_cov_num(){
    int hyp_num;
    int Q = kernel_param[0];
    int D = kernel_param[1];
    int R = kernel_param[2];
    switch(kernel_index){
        case 0:
            hyp_num = 2;
            break;
        case 7:
            hyp_num = Q*(D*R + 2 + D);
            break;
        case 8:
            hyp_num = 3*Q;
            break;
        default:
            cout << "ERROR: unknown mode for getting covariance parameter for kernel (" 
                 << kernel_index << ")" << endl;
            exit(1);
            break;
    }
    return hyp_num;
}

int c_experiment::get_test_cov_num(int fold, const string &kernel_clust_alg){
    int hyp_num;

    vector<int> test_kernel_param;
    test_kernel_param = get_test_kernel_param(fold, kernel_clust_alg);

    int Q = test_kernel_param[0];
    int D = test_kernel_param[1];
    int R = test_kernel_param[2];

    switch(kernel_index){
        case 0:
            hyp_num = 2;
            break;
        case 7:
            hyp_num = Q*(D*R + 2 + D);
            break;
        case 8:
            hyp_num = 3*Q;
            break;
        default:
            cout << "ERROR: unknown mode for getting covariance parameter for kernel (" 
                 << kernel_index << ")" << endl;
            exit(1);
            break;
    }
    return hyp_num;
}

int c_experiment::get_lik_num(){
    int hyp_num;
    switch(kernel_index){
        case 0:
            hyp_num = 1;
            break;
        case 7:
            hyp_num = kernel_param[1];
            break;
        case 8:
            hyp_num = 1;
            break;
        default:
            cout << "ERROR: unknown mode for getting covariance parameter for kernel (" 
                 << kernel_index << ")" << endl;
            exit(1);
            break;
    }
    return hyp_num;
}

int c_experiment::get_mean_num(){
    // support only zero mean function now
    int hyp_num = 0;
    return hyp_num;
}

void c_experiment::get_hyp_bounds(){
    int     hyp_num;
    double  temp;

    // get the number of hyperparameters
    hyp_num = get_hyp_num();
    ifstream data(exp_hyp_bound_file.c_str(), ios::in);

    if(!data){
        cerr << "File " << exp_hyp_bound_file << " could not be opened." << endl;
        exit(1);
    }
    hyp_array_ub.clear();
    hyp_array_lb.clear();
    for(int i = 0; i < hyp_num; i++){
        data >> temp;
        hyp_array_lb.push_back(temp);
        data >> temp;
        hyp_array_ub.push_back(temp);
    }
    data.close();

}
void c_experiment::get_global_hyp(vector< vector<double> > &global_hyp_array){
    cout << "generating random hyperparameters..." << endl;
    srand(srand_seed);

    for(int i = 0; i < scg_init_num; i++){
        vector<double> hyp_array;
        switch(kernel_index){
            case 0:
                get_hyp_SE(hyp_array);
                break;
            case 7:
                get_hyp_LMC_SM(hyp_array);
                break;
            case 8:
                get_hyp_SM(hyp_array);
                break;
            default:
                cout << "ERROR: unknown mode for kernel (" << kernel_index << ")" << endl;
                exit(1);
                break;
        }
        global_hyp_array.push_back(hyp_array);
    }
}

void c_experiment::print_experiment(){
    string kernel_name;
    cout << "---------------------------------------------------" << endl;
    cout << "Summary of the experiment:" << endl;
    cout << "Input config. file: "      << exp_cfg_file << endl;
    cout << "Input data path: "         << exp_data_dir << endl;
    cout << "Training result output path: " << exp_train_dir << endl;
    cout << "Testing result output path: " << exp_test_dir << endl;
    cout << "Index of testing feature(s): ";
    for(int i = 0; i < int(feature_index.size()); i++){
        cout << feature_index[i] << ' ';
    }
    cout << endl;
    
    cout << "Total # of CV fold: " << cv_fold_num << endl;
    cout << endl;

    cout << "Current kernel index: " << kernel_index << endl;
    cout << "Current random seed: " << srand_seed << endl;
    cout << endl;

    cout << "Loading hyperparameters boundary from: " << exp_hyp_bound_file << endl;
    cout << "Total number of hyperparameters: " << get_hyp_num() << endl;
    cout << "---------------------------------------------------" << endl;

}

void c_experiment::output_double_bin(const string &file_prefix, vector<double> hyp_array){
    string out_file = file_prefix + ".bin";
    ofstream data(out_file.c_str(), ios::binary);
    if(data.is_open()) {
        for(int i = 0; i < int(hyp_array.size()); i++){
            data.write(reinterpret_cast<char*>(&hyp_array[i]), sizeof(double));
        }
    }
    data.close();
}

void c_experiment::output_int_txt(const string &file_prefix, vector<int> int_array){
    string out_file = file_prefix + ".txt";
    ofstream data(out_file.c_str(), ios::out);
    if(data.is_open()){
        for(int i = 0; i < int(int_array.size()); i++){
            data << int_array[i];
            data << "\n";
        }
    }
    data.close();
}

double c_experiment::get_one_random(
                                    const double &lb,
                                    const double &ub,
                                    const double &scale, 
                                    const bool &flag_inv,
                                    const bool &flag_log
                                    ){
    double a, temp;
    int digit = 12;
    int rand_max = floor(pow(2.0, digit));

    temp = ((double)(rand() % rand_max)) + 1.0;
    temp *= (ub - lb);
    temp = temp/((double) rand_max);

    a = scale*(temp + lb);
    if(flag_inv){
        a = 1.0 / a;
    }

    if(flag_log){
        a = log(a);
    }
    return a;
}

void c_experiment::get_hyp_SE(vector<double> &hyp_array){
    double temp;
    for(int i = 0; i < get_hyp_num(); i++){
        if(i < (get_lik_num() + get_cov_num())){
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, true);
        }
        else{
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, false);
        }
        hyp_array.push_back(temp);
    }
}

void c_experiment::get_hyp_LMC_SM(vector<double> &hyp_array){
    double temp;
    int Q = kernel_param[0];
    int D = kernel_param[1];
    int R = kernel_param[2];
    for(int i = 0; i < get_hyp_num(); i++){
        if(i < get_lik_num()){
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, true);
        }
        else if(i < (get_lik_num() + Q*D*R)){
            double dQ, dR;
            dQ = Q;
            dR = R;
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 0.9/sqrt(dQ*dR), false, false);
        }
        else if( i < (get_lik_num() + Q*(D*R+1)) ){ // mu
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, false);
            temp = log(1.0/temp);
        }
        else if( i < (get_lik_num() + Q*(D*R+2)) ){ // v
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, false);
            temp = log(1.0/(2*PI*temp));
        }
        else if( i < (get_lik_num() + Q*(D*R+2 + D)) ){
            double dQ = Q;
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 0.1/dQ, false, true);
        }
        else{
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, false);
        }
        hyp_array.push_back(temp);
    }
}

void c_experiment::get_hyp_SM(vector<double> &hyp_array){
    double temp;
    int Q = kernel_param[0];
    for(int i = 0; i < get_hyp_num(); i++){
        if(i < get_lik_num()){
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, true);  
        }
        else if(i < (get_lik_num() + Q)){
            double dQ = Q;
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0/dQ, false, true);   
        }
        else if( i < (get_lik_num() + 2*Q) ){
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, true, true);
        }
        else if( i < (get_lik_num() + 3*Q) ){
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 2*PI, true, true);
        }
        else{
            temp = get_one_random(hyp_array_lb[i], hyp_array_ub[i], 1.0, false, false);
        }
        hyp_array.push_back(temp);
    }
}


