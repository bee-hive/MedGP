/*
-------------------------------------------------------------------------
This is the header file for Gaussian process regression to include
all kernels, mean functions, likelihood functions and inference functions.
-------------------------------------------------------------------------
*/
#include "util/global_settings.h"

#include "kernel/c_kernel.h"
#include "kernel/c_kernel_SE.h"
#include "kernel/c_kernel_SM.h"
#include "kernel/c_kernel_LMC_SM.h"

#include "mean/c_meanfunc.h"
#include "mean/c_meanfunc_zero.h"

#include "likelihoods/c_likelihood.h"
#include "likelihoods/c_likelihood_gaussian.h"
#include "likelihoods/c_likelihood_gaussianMO.h"

#include "inference/c_inference.h"
#include "inference/c_inference_exact.h"
#include "inference/c_inference_prior.h"

#include "prior/c_prior.h"
