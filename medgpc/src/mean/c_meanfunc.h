/*
-------------------------------------------------------------------------
This is the header file for the class mean function.
All other mean functions should inherit this class.
-------------------------------------------------------------------------
*/

#ifndef C_MEANFUNC_H
#define C_MEANFUNC_H
# include <vector>
# include <string>

using namespace std;

class c_meanfunc{

    public:
        c_meanfunc();
        c_meanfunc(vector<int> input_param);
        c_meanfunc(vector<int> input_param, vector<double> input_hyp);
        virtual void compute_mean_vector(
                                            const vector<int> &meta, 
                                            const vector<float> &x, 
                                            const bool &flag_grad, 
                                            float *&mean_vector, 
                                            vector<float*> &mean_gradients
                                            ){};
        virtual void compute_mean_gradients(
                                            const vector<int> &meta, 
                                            const vector<float> &x,
                                            const float *chol_alpha,
                                            vector<double> &gradients
                                            ){};

        void print_meanfunc();
        void set_meanfunc_hyp(vector<double> input_hyp);
        void set_meanfunc_param(vector<int> input_param);
        
        vector<int>     get_meanfunc_param();
        vector<double>  get_meanfunc_hyp(); // for debugging
        int             get_meanfunc_hyp_num();

    protected:
        vector<double>  meanfunc_hyp;
        vector<int>     meanfunc_param;
        int             meanfunc_hyp_num;
        string          meanfunc_name;

};
#endif
