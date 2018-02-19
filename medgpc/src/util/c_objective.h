/*
-------------------------------------------------------------------------
This is the header file for the class objective.
Different objective functions should inherit this class.
-------------------------------------------------------------------------
*/
#ifndef C_OBJECTIVE_H
#define C_OBJECTIVE_H
#include <iostream>
#include <vector>
#include <string>

#include "core/gp_model_include.h"
#include "core/gp_regression.h"

using namespace std;

class c_objective{

    public:
        c_objective(){
            objective_name = "c_objective";
            dist_thread_num = 1;
        };
        void print_objective(){
            cout << "current objective: " << objective_name << endl;
        };
        virtual void set_dist_thread_num(int input_thread_num){};
        virtual bool compute_objective(
                                        const bool &flag_grad, 
                                        const vector<double> &input_parameter, 
                                        double &objective_value, 
                                        vector<double> &gradients,
                                        c_kernel        *&input_kernel, 
                                        c_meanfunc      *&input_meanfunc, 
                                        c_likelihood    *&input_likfunc, 
                                        c_inference     *&input_inffunc,
                                        c_prior         *&input_prior
                                        ){};
    protected:
        string  objective_name;
        int     dist_thread_num;

};
#endif