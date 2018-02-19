/*
-------------------------------------------------------------------------
This is the header file for the class likelihood.
Gaussian likelihood can be put into inference class directly,
but all other class should inherit this class.
-------------------------------------------------------------------------
*/
#ifndef C_LIKELIHOOD_GAUSSIANMO_H
#define C_LIKELIHOOD_GAUSSIANMO_H
#include <vector>
#include <string>
#include "likelihoods/c_likelihood.h"

using namespace std;

class c_likelihood_gaussianMO:public c_likelihood{

    public:
        c_likelihood_gaussianMO();
        c_likelihood_gaussianMO(vector<int> input_param);
        c_likelihood_gaussianMO(vector<int> input_param, vector<double> input_hyp);
        void compute_lik_vector(
                                const vector<int> &meta, 
                                const vector<float> &x, 
                                const bool &flag_grad, 
                                float *&lik_vector, 
                                vector<float*> &lik_gradients
                                );

};
#endif