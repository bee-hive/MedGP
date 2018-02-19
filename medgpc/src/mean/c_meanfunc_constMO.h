/*
-------------------------------------------------------------------------
This is the header file for the class const mean function for multi-output.
-------------------------------------------------------------------------
*/
#ifndef C_MEANFUNC_CONSTMO_H
#define C_MEANFUNC_CONSTMO_H
# include <vector>
# include "mean/c_meanfunc.h"

using namespace std;

class c_meanfunc_constMO:public c_meanfunc{

    public:
        c_meanfunc_constMO();
        c_meanfunc_constMO(vector<int> input_param);
        c_meanfunc_constMO(vector<int> input_param, vector<double> input_hyp);
        void compute_mean_vector(
                                    const vector<int> &meta, 
                                    const vector<float> &x, 
                                    const bool &flag_grad, 
                                    float *&mean_vector, 
                                    vector<float*> &mean_gradients
                                    );
        void compute_mean_gradients(
                                    const vector<int> &meta, 
                                    const vector<float> &x,
                                    const float *chol_alpha,
                                    vector<double> &gradients
                                    );
};
#endif
