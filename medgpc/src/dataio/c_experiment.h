/*
-------------------------------------------------------------------------
This is the header file for class of experiment.
-------------------------------------------------------------------------
*/
#ifndef C_EXPERIMENT_H
#define C_EXPERIMENT_H
#include <vector>
#include <string>

#include "core/gp_model_include.h"
using namespace std;

class c_experiment{
    
    public:
        c_experiment();
        c_experiment(const string &input_cfg_name);
        void init_default_param();

        string          get_exp_train_dir();
        string          get_exp_test_dir();
        int             get_kernel_index();
        vector<int>     get_kernel_param();
        vector<int>     get_feature_index();

        vector<int>     get_lik_param();
        int             get_prior_mode();

        int             get_cv_fold_num();
        int             get_scg_init_num();
        int             get_scg_max_iter_num();
        double          get_online_learn_rate();
        double          get_online_momentum();

        vector<float>   get_prior_hyp();
        void get_one_patient_data(
                                    string PAN,
                                    vector<int> &meta_vec,
                                    vector<float> &time_vec, 
                                    vector<float> &value_vec
                                    );

        // get lower/upper bounds for hyperparamters
        int     get_hyp_num();
        int     get_cov_num();
        int     get_lik_num();
        int     get_mean_num();
        void    get_hyp_bounds();
        void    get_global_hyp(vector< vector<double> > &global_hyp_array);
        int     get_prior_sub_opt_iter();
        
        // get random initializations
        double  get_one_random(
                                const double &lb,
                                const double &ub,
                                const double &scale, 
                                const bool &flag_inv,
                                const bool &flag_log
                                );
        void    get_hyp_SE(vector<double> &hyp_array);
        void    get_hyp_SM(vector<double> &hyp_array);
        void    get_hyp_LMC_SM(vector<double> &hyp_array);
        void    print_experiment();

        void    output_double_bin(const string &file_prefix, vector<double> hyp_array);
        void    output_int_txt(const string &file_prefix, vector<int> int_array);

        vector<int>     get_test_kernel_param(int fold, const string &kernel_clust_alg);
        int             get_test_cov_num(int fold, const string &kernel_clust_alg);
        vector<double>  get_test_mode_param(int fold, const string &kernel_clust_alg);

    private:
        string      experiment_name;
        string      exp_cfg_file;
        string      exp_data_dir;
        string      exp_hyp_bound_file;

        string      exp_top_dir;
        string      exp_train_dir;
        string      exp_test_dir;
        string      exp_kernel_dir;
        
        int         srand_seed;
        int         kernel_index;
        vector<int> kernel_param;
        vector<int> feature_index;
        int         cv_fold_num;
        int         scg_init_num;
        int         scg_max_iter_num;
        int         prior_mode;
        int         prior_sub_opt_iter;

        double      learn_rate;
        double      momentum;

        vector<float>   prior_hyp;

        vector<double>  hyp_array_ub;
        vector<double>  hyp_array_lb;   

};
#endif