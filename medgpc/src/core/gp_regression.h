/*
-------------------------------------------------------------------------
This is the header file for class Gaussian process regression.
-------------------------------------------------------------------------
*/
#ifndef GP_REGRESSION_H
#define GP_REGRESSION_H
#include <vector>
#include "core/gp_model_include.h"

using namespace std;

class GP_Regression{

    public:
        GP_Regression();
        GP_Regression(
                        const int &input_dim, 
                        c_kernel        *input_kernel, 
                        c_meanfunc      *input_meanfunc, 
                        c_likelihood    *input_likfunc, 
                        c_inference     *input_inffunc,
                        c_prior         *input_prior
                        );
        ~GP_Regression();

        void    reset(
                        const int &input_dim, 
                        c_kernel        *input_kernel, 
                        c_meanfunc      *input_meanfunc, 
                        c_likelihood    *input_likfunc, 
                        c_inference     *input_inffunc,
                        c_prior         *input_prior
                        );
        void    set_kernel(c_kernel *input_c_kernel);
        void    set_meanfunc(c_meanfunc *input_meanfunc);
        void    set_likelihood(c_likelihood *input_likfunc);
        void    set_inference(c_inference *input_inffunc);
        void    set_prior(c_prior *input_prior);

        int                 get_dim();
        bool                get_flag_trained();
        double              get_neg_log_mlikelihood();
        vector<double>      get_dneg_log_mlikelihood();
        void                set_neg_log_mlikelihood(const double &input_nlml);
        void                set_dneg_log_mlikelihood(const vector<double> &input_dnlml);

        void                        train(
                                            const bool &flag_grad, 
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            const vector<float> &y
                                            );
        vector< vector<float> >     predict(
                                            const vector<int> &meta, 
                                            const vector<int> &meta2, 
                                            const vector<float> &x, 
                                            const vector<float> &y, 
                                            const vector<float> &x2
                                            );
        // for debugging
        vector< vector<float> >     parsed_predict(
                                            const int &meta_num,
                                            const vector<int> &meta, 
                                            const vector<int> &meta2, 
                                            const vector<float> &x, 
                                            const vector<float> &y, 
                                            const vector<float> &x2
                                            );
        

    private:
        int                     dim;
        int                     inf_thread_num;
        bool                    flag_trained;
        double                  nlm_likelihood;     // negative log marginal likelihood
        vector<double>          dnlm_likelihood;    // gradients w.r.t each hyperparameters
        float                   *chol_factor_inv;
        float                   *chol_alpha;
        float                   beta;

        c_kernel                *kernel;
        c_meanfunc              *meanfunc;
        c_likelihood            *likfunc;
        c_inference             *inffunc;
        c_prior                 *prior;

};
#endif

