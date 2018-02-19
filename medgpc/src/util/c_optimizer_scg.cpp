/*
-------------------------------------------------------------------------
This is the function file for the class scaled conjugate gradient descent.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>

#include "util/c_optimizer_scg.h"
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

using namespace std;

void c_optimizer_scg::optimize(
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
                                ){
    double INT = 0.1;
    double EXT = 3.0;
    double MAX = 20;
    double RATIO = 10;
    double SIG = 0.1;
    double RHO = SIG/2.0;

    double red = 1.0;
    string method;

    if(max_iteration > 0){
        method = "Linesearch ";
    }
    else{
        method = "Function evaluation ";
    }

    int i = 0;
    bool ls_failed = false;
    double *s;
    double d0, f0;
    double x1, x2, x3, x4, d1, d2, d3, d4, f1, f2, f3, f4;
    double F0, M, A, B;
    vector<double> dF0, X0;
    vector<double> df0, df3;
    bool obj_flag;

    // compute the objective using initial guess
    obj_flag = objfunc -> compute_objective(
                                            true, init_parameter, f0, df0, 
                                            input_kernel, input_meanfunc, 
                                            input_likfunc, input_inffunc, 
                                            input_prior);
    if(display)
        cout << method << i << ": " << f0 << endl;
    
    i = i + signbit(max_iteration);
    
    s = new double[int(df0.size())];
    for(int j = 0; j < int(df0.size()); j++){
        s[j] = -1.0*df0[j];
    }
    MKL_INT n = int(df0.size());
    d0 = cblas_ddot(n, s, 1, s, 1);
    d0 = (-1.0) * d0;
    x3 = red/(1.0 - d0);

    // begin the searching loop
    opt_loss = f0; // fX, f0(in the while loop) --> opt_loss
    opt_parameter = init_parameter; // X --> opt_parameter
    while( i < abs(max_iteration) ){
        i = i + signbit(max_iteration);

        X0 = opt_parameter;
        F0 = opt_loss;
        dF0 = df0;

        if(max_iteration > 0){
            M = MAX;
        }
        else{
            M = min((int)MAX, abs(max_iteration)-i);
        }

        while(1){
            x2 = 0.0;
            
            f2 = opt_loss;
            d2 = d0;

            f3 = opt_loss;
            df3 = df0;

            bool success = false;

            while((!success) && (M > 0)){
                M = M - 1;
                i = i + signbit(max_iteration);
                
                vector<double> new_parameter;
                for(int j = 0; j < int(opt_parameter.size()); j++){
                    new_parameter.push_back( opt_parameter[j] + x3*s[j] );
                }
                obj_flag = objfunc -> compute_objective(true, new_parameter, f3, df3, 
                                                        input_kernel, input_meanfunc, 
                                                        input_likfunc, input_inffunc,
                                                        input_prior);

                if( (!obj_flag) || std::isinf(f3) || std::isnan(f3) ){
                    x3 = (x2 + x3)/2.0;
                }
                else{
                    success = true;
                }
            }// while((success == false) && (M > 0))

            if(f3 < F0){
                for(int j = 0; j < int(opt_parameter.size()); j++){
                    X0[j] = opt_parameter[j] + x3*s[j];
                }
                F0 = f3;
                dF0 = df3;
            }

            double *a;
            a = new double[int(df3.size())];
            for(int ii = 0; ii < int(df3.size()); ii++){
                a[ii] = df3[ii];
            }
            d3 = cblas_ddot(n, a, 1, s, 1);
            delete[] a;
            

            if( (d3 > SIG*d0) || (f3 > (opt_loss + x3*RHO*d0)) || (M == 0)){
                break;
            }
                

            x1 = x2;
            f1 = f2;
            d1 = d2;

            x2 = x3;
            f2 = f3;
            d2 = d3;

            A = 6.0*(f1-f2) + 3.0*(d2+d1)*(x2-x1);
            B = 3.0*(f2-f1) - (2.0*d1+d2)*(x2-x1);
            double temp = B*B - A*d1*(x2-x1);
            if( temp < 0 ){
                x3 = x2*EXT;
            }
            else{
                x3 = x1 - (d1*pow(x2-x1, 2.0) / (B + sqrt(temp)));
                if(std::isnan(x3) || std::isinf(x3) || (x3 < 0)){
                    x3 = x2*EXT;
                }
                else if(x3 > x2*EXT){
                    x3 = x2*EXT;
                }
                else if( x3 < (x2+INT*(x2-x1)) ){
                    x3 = x2 + INT*(x2-x1);
                }
            }
        }// while(1)

        while( ( (fabs(d3) > -1.0*SIG*d0) || (f3 > (opt_loss + x3*RHO*d0)) ) && (M > 0) ){
            if( (d3 > 0) || (f3 > (opt_loss + x3*RHO*d0)) ){
                x4 = x3;
                f4 = f3;
                d4 = d3;
            }   
            else{
                x2 = x3;
                f2 = f3;
                d2 = d3;
            }

            if(f4 > opt_loss){
                x3 = x2 - (0.5*d2*pow(x4-x2, 2.0))/(f4-f2-d2*(x4-x2));
                if(std::isnan(x3) || std::isinf(x3)){
                    x3 = (x2+x4)/2.0;   
                }
            }
            else{
                A = 6.0*(f2-f4)/(x4-x2) + 3.0*(d4+d2);
                B = 3.0*(f4-f2) - (2.0*d2 + d4)*(x4-x2);

                if( (B*B - A*d2*pow(x4-x2, 2.0)) < 0){
                    x3 = (x2+x4)/2.0;
                }
                else{
                    x3 = x2 + (sqrt(B*B - A*d2*pow(x4-x2, 2.0)) - B)/A;
                    if(std::isnan(x3) || std::isinf(x3)){
                        x3 = (x2+x4)/2.0;   
                    }
                }
            }
            x3 = max(min(x3, x4-INT*(x4-x2)), x2+INT*(x4-x2));

            vector<double> new_parameter;
            for(int j = 0; j < int(opt_parameter.size()); j++){
                new_parameter.push_back( opt_parameter[j] + x3*s[j] );
            }
            obj_flag = objfunc -> compute_objective(true, new_parameter, f3, df3, 
                                                    input_kernel, input_meanfunc, 
                                                    input_likfunc, input_inffunc,
                                                    input_prior);

            if((obj_flag) && (f3 < F0)){
                for(int j = 0; j < int(opt_parameter.size()); j++){
                    X0[j] = opt_parameter[j] + x3*s[j];
                }
                F0 = f3;
                dF0 = df3;
            }
            M = M - 1;
            i = i + signbit(max_iteration);

            double *a = &df3[0];
            d3 = cblas_ddot(n, a, 1, s, 1);
        }

        if((obj_flag) && (fabs(d3) < -1.0*SIG*d0) && (f3 < (opt_loss + x3*RHO*d0))){
            for(int j = 0; j < int(opt_parameter.size()); j++){
                opt_parameter[j] = opt_parameter[j] + x3*s[j];
            }
            opt_loss = f3;
            if(display)
                cout << method << i << ": " << opt_loss << endl;

            double *a = &df3[0];
            double *b = &df0[0];
            double df3_df3 = cblas_ddot(n, a, 1, a, 1);
            double df3_df0 = cblas_ddot(n, a, 1, b, 1);
            double df0_df0 = cblas_ddot(n, b, 1, b, 1);
            for(int j = 0; j < int(df3.size()); j++){
                s[j] = ((df3_df3 - df3_df0)/df0_df0)*s[j] - df3[j];
            }
            df0 = df3;
            d3 = d0;
            d0 = cblas_ddot(n, b, 1, s, 1);

            if(d0 > 0){
                for(int j = 0; j < int(df0.size()); j++){
                    s[j] = -1.0*df0[j];
                }
                d0 = -1.0*cblas_ddot(n, s, 1, s, 1);
            }            
            x3 = x3 * min(RATIO, d3/(d0 - pow(2.0, -52)));
            ls_failed = false;
        }
        else{
            opt_parameter = X0;
            opt_loss = F0;
            df0 = dF0;

            for(int j = 0; j < int(df0.size()); j++){
                s[j] = -1.0*df0[j];
            }
            d0 = -1.0*cblas_ddot(n, s, 1, s, 1);
            x3 = 1.0/(1.0 - d0);
            ls_failed = true;
        }
    }// while(i < abs(max_iteration))
    
    delete[] s;
}
