/*
-------------------------------------------------------------------------
This is the header file for the class hyperparameters.
-------------------------------------------------------------------------
*/
#ifndef C_HYPERPARAM_H
#define C_HYPERPARAM_H
#include <vector>
#include <string>

using namespace std;

class c_hyperparam{

    public:
        c_hyperparam();
        c_hyperparam(const c_hyperparam &hyp);
        c_hyperparam(
                        const vector<double> &hyp_all, 
                        const int &num_cov, 
                        const int &num_mean, 
                        const int &num_lik
                        );
        c_hyperparam(
                        const vector<double> &hyp_cov, 
                        const vector<double> &hyp_mean, 
                        const vector<double> &hyp_lik
                        );
        
        int get_num_hyp_cov();
        int get_num_hyp_mean();
        int get_num_hyp_lik();
        int get_num_hyp_all();

        vector<double> get_hyp_cov();
        vector<double> get_hyp_mean();
        vector<double> get_hyp_lik();
        vector<double> get_hyp_all();

        void set_hyp_cov(const vector<double> &input_hyp_cov);
        void set_hyp_mean(const vector<double> &input_hyp_mean);
        void set_hyp_lik(const vector<double> &input_hyp_lik);
        void set_hyp_all(
                            const vector<double> &input_hyp, 
                            const int &num_cov, 
                            const int &num_mean, 
                            const int &num_lik
                            );

    private:
        vector<double> hyp_cov;
        vector<double> hyp_mean;
        vector<double> hyp_lik;

};
#endif
