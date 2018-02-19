/*
-------------------------------------------------------------------------
This is the header file for the class optimizer.
Different optimizers should inherit this class.
-------------------------------------------------------------------------
*/
#ifndef C_OPTIMIZER_H
#define C_OPTIMIZER_H
#include <iostream>
#include <vector>
#include <string>

#include "util/c_objective.h"
#include "core/gp_model_include.h"

using namespace std;

class c_optimizer{

    public:
        c_optimizer(){
            optimizer_name = "c_optimizer";
        };
        void print_optimizer(){
            cout << "current optimizer: " << optimizer_name << endl;
        };
        virtual void optimize(
                                const int &max_iteration,
                                const vector<double> &init_parameter,
                                c_objective *objfunc,
                                const bool &display,
                                double &opt_loss, vector<double> &opt_parameter,
                                c_kernel        *&input_kernel, 
                                c_meanfunc      *&input_meanfunc, 
                                c_likelihood    *&input_likfunc, 
                                c_inference     *&input_inffunc,
                                c_prior         *&input_prior
                                ){};

    protected:
        string  optimizer_name;

};
#endif