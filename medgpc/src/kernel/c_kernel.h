/*
-------------------------------------------------------------------------
This is the header file for top kernel class.
All other kernels should inherit this class.
-------------------------------------------------------------------------
*/
#ifndef C_KERNEL_H
#define C_KERNEL_H
#include <vector>
#include <string>

using namespace std;


class c_kernel{

    public:
        c_kernel();
        c_kernel(const vector<int> &input_param);
        c_kernel(const vector<int> &input_param, const vector<double> &input_hyp);
        virtual void compute_self_diag_matrix(
                                                const vector<int> &meta, 
                                                const vector<float> &x, 
                                                float *&diag_gram_matrix
                                                ){};
        virtual void compute_self_gram_matrix(
                                                const vector<int> &meta, 
                                                const vector<float> &x,
                                                float *&self_gram_matrix
                                                ){};
        virtual void compute_self_gradients(
                                            const vector<int> &meta, 
                                            const vector<float> &x,
                                            const float *chol_Q,
                                            vector<double> &gradients
                                            ){};

        virtual void compute_cross_gram_matrix(
                                                const vector<int> &meta, 
                                                const vector<int> &meta2, 
                                                const vector<float> &x, 
                                                const vector<float> &x2, 
                                                float *&cross_gram_matrix
                                                ){};

        virtual void reset_coregional_matrix(vector< vector<double> > em_B_array){};
        
        void compute_squared_dist(
                                    const vector<float> &x, 
                                    const vector<float> &x2, 
                                    float *&dist_matrix
                                    );
        void compute_abs_dist(
                                const vector<float> &x, 
                                const vector<float> &x2, 
                                float *&dist_matrix
                                );
        void compute_sin_dist(
                                double period, 
                                const vector<float> &x, 
                                const vector<float> &x2, 
                                float *&dist_matrix
                                );
        void compute_scale_abs_dist(
                                double lengthscale, 
                                const vector<float> &x, 
                                const vector<float> &x2, 
                                float *&dist_matrix
                                );
        void compute_scale_squared_dist(
                                        double lengthscale, 
                                        const vector<float> &x, 
                                        const vector<float> &x2, 
                                        float *&dist_matrix
                                        );
        void compute_meta_map(
                                int dim, 
                                const vector<double> &coregional_matrix, 
                                const vector<int> &meta, 
                                const vector<int> &meta2, 
                                vector<float> &map_matrix
                                );
        void print_kernel();
        void set_kernel_grad_thread(int input_thread_num);
        int get_kernel_grad_thread();
        virtual void set_kernel_hyp(const vector<double> &input_hyp){};
        virtual void set_kernel_param(const vector<int> &input_param){};
        
        int             get_kernel_hyp_num();
        vector<int>     get_kernel_param();
        vector<double>  get_kernel_hyp();

    protected:
        vector<double>  kernel_hyp;
        vector<int>     kernel_param;
        int             kernel_hyp_num;
        int             kernel_grad_thread;
        string          kernel_name;
        // at least 2 elements here:
        // (1) dim_num;
        // (2) type_num;
        // for other kernels such as spectral mixture, ICM, LMC,
        // define other elements in each sub-class.

};
#endif