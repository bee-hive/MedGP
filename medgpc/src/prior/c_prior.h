/*
-------------------------------------------------------------------------
This is the header file for the class priors.
-------------------------------------------------------------------------
*/
#ifndef C_PRIOR_H
#define C_PRIOR_H
#include <vector>

using namespace std;

class c_prior{

    public:
        c_prior();
        c_prior(int num_cov, int num_mean, int num_lik);
        c_prior(const c_prior &prior);

        void initialize_param(  
                                int num_cov, 
                                int num_mean, 
                                int num_lik
                                );
        void setup_param(
                            const int kernel_index, 
                            const vector<int> &kernel_param, 
                            const int &mode,
                            const vector<float> &prior_param
                            );
        void setup_hier_gamma_prior(
                                    const vector<int> &kernel_param,
                                    const vector<float> &prior_param
                                    );

        // flag to indicate if a hyperparameter is still active
        vector<bool>    flag_cov;
        vector<bool>    flag_mean;
        vector<bool>    flag_lik;

        // flag to indicate if a hyperparameted is using exponential transform
        vector<bool>    exp_cov;
        vector<bool>    exp_mean;
        vector<bool>    exp_lik;

        // the vector of parameter for priors
        vector< vector<float> > fix_param_cov;
        vector< vector<float> > fix_param_mean;
        vector< vector<float> > fix_param_lik;

        // prior type code: 0: Clamp, 1: Normal, 2: Laplace
        vector<int>     type_cov;
        vector<int>     type_mean;
        vector<int>     type_lik;

        vector<double> get_one_lik_cov(const double &x, const int &index);
        vector<double> get_one_lik_lik(const double &x, const int &index);
        vector<double> get_one_lik_mean(const double &x, const int &index);
        vector<double> prior_lik_normal(const double &x, const vector<float> &param);
        vector<double> prior_lik_laplace(const double &x, const vector<float> &param);
        vector<double> prior_lik_kde(const double &x, const vector<float> &param);

        void init_cov_varEM(int num_cov_varEM, double init_val);
        void init_cov_varEM_fix(int num_cov_varEM_fix, double init_val);

        void set_cov_varEM_all(const vector<double> &input_varEM);
        void set_cov_varEM_one(double value, const int &index);
        vector<double> get_cov_varEM_all();
        double get_cov_varEM_one(const int &index);

        void set_cov_varEM_fix_all(const vector<double> &input_varEM_fix);
        void set_cov_varEM_fix_one(double value, const int &index);
        vector<double> get_cov_varEM_fix_all();
        double get_cov_varEM_fix_one(const int &index);
        
        // for testing
        void    init_test_prior(  const int kernel_index, 
                                  const vector<int> &test_kernel_param,
                                  const vector<double> &test_mode_param
                                  );
        bool    get_one_prior_flag(const int &index);
        int     get_one_prior_type(const int &index);

        // debugging
        void print_status();
        void print_one_prior(
                            const bool &flag, 
                            const int &type, 
                            const vector<float> &param
                            );

    private:
        int hyp_cov_num;
        int hyp_mean_num;
        int hyp_lik_num;
        vector<double>  cov_varEM_fix;
        vector<double>  cov_varEM;
        vector<double>  test_mode_parameter;

};
#endif
