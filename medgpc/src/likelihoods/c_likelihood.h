/*
-------------------------------------------------------------------------
This is the header file for the class likelihood.
Gaussian likelihood can be put into inference class directly,
but all other class should inherit this class.
-------------------------------------------------------------------------
*/
#ifndef C_LIKELIHOOD_H
#define C_LIKELIHOOD_H
#include <vector>
#include <string>

using namespace std;

class c_likelihood{

    public:
        c_likelihood();
        c_likelihood(vector<int> input_param);
        c_likelihood(vector<int> input_param, vector<double> input_hyp);
        virtual void compute_lik_vector(
                                        const vector<int> &meta, 
                                        const vector<float> &x, 
                                        const bool &flag_grad, 
                                        float *&lik_vector, 
                                        vector<float*> &lik_gradients
                                        ){};

        void print_likfunc();
        void set_likfunc_hyp(vector<double> input_hyp);
        void set_likfunc_param(vector<int> input_param);
        
        vector<int>     get_likfunc_param();
        vector<double>  get_likfunc_hyp();  // for debugging
        int         get_likfunc_hyp_num();  // for debugging

    protected:
        vector<double>  likfunc_hyp;
        vector<int>     likfunc_param;
        int             likfunc_hyp_num;
        string          likfunc_name;

};
#endif
