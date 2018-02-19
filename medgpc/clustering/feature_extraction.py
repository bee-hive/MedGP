import numpy as np
from ..visualization.fastkernel import compute_B_matrix, compute_sm_1d


def extract_kernel_feature(kernel_type, Q, D, R, pan_array, hyp_array):
    if(kernel_type == 'SE'): # SE
        comp_pan, comp_qidx, comp_feature = extract_SE_feature(pan_array, hyp_array)
    elif(kernel_type == 'SM'): # SM
        comp_pan, comp_qidx, comp_feature = extract_SM_feature(pan_array, hyp_array, Q)
    elif(kernel_type == 'LMC-SM'): # LMC-SM
        comp_pan, comp_qidx, comp_feature = extract_LMC_SM_feature(pan_array, hyp_array, Q, D, R)
    else:
        print('specified kernel type {} not supported'.format(kernel_type))
        raise NotImplementedError
    return comp_pan, comp_qidx, comp_feature


def extract_SE_feature(pan_array, hyp_array):
    assert hyp_array.shape[1] == 3
    scale_thr = np.power(10., -10.)
    comp_pan = []
    comp_qidx = []
    comp_feature = []
    for i in range(len(pan_array)):
        pan = pan_array[i]
        hyp = hyp_array[i, :]
        sf2 = np.exp(2*hyp[2])
        if(abs(sf2) > scale_thr):
            comp_pan.append(pan)
            comp_qidx.append(0)
            comp_feature.append(np.exp(hyp[1])) # lengthscale 
    comp_pan = np.asarray(comp_pan)
    comp_qidx = np.asarray(comp_qidx)
    comp_feature = np.asarray(comp_feature)
    return comp_pan, comp_qidx, comp_feature


def extract_SM_feature(pan_array, hyp_array, Q):
    assert hyp_array.shape[1] == (3*Q+1)
    scale_thr = np.power(10., -10.)
    comp_pan = []
    comp_qidx = []
    comp_feature = []
    for i in range(len(pan_array)):
        pan = pan_array[i]
        hyp = hyp_array[i, :]
        for q in range(Q):
            sf2 = np.exp(hyp[1 + q])
            if(abs(sf2) > scale_thr):
                mu = np.exp(hyp[1 + Q + q])
                v = np.exp(2*hyp[1 + 2*Q + q])
                resp = compute_sm_feature(mu, v)
                comp_pan.append(pan)
                comp_qidx.append(q)
                comp_feature.append(resp)
    comp_pan = np.asarray(comp_pan)
    comp_qidx = np.asarray(comp_qidx)
    comp_feature = np.asarray(comp_feature)
    return comp_pan, comp_qidx, comp_feature


def extract_LMC_SM_feature(pan_array, hyp_array, Q, D, R):
    assert hyp_array.shape[1] == (D + Q*(D*R+2+D))
    scale_thr = np.power(10., -10.)    
    comp_pan = []
    comp_qidx = []
    comp_feature = []
    for i in range(len(pan_array)):
        pan = pan_array[i]
        hyp = hyp_array[i, :]

        B = compute_B_matrix(Q, D, R, hyp)
        for q in range(Q):
            if(np.max(abs(B[q].flatten())) > scale_thr):
                mu = np.exp(hyp[D + Q*D*R + q])
                v = np.exp(2*hyp[D + Q*(D*R + 1) + q])
                resp = compute_sm_feature(mu, v)
                comp_pan.append(pan)
                comp_qidx.append(q)
                comp_feature.append(resp)
    comp_pan = np.asarray(comp_pan)
    comp_qidx = np.asarray(comp_qidx)
    comp_feature = np.asarray(comp_feature)
    return comp_pan, comp_qidx, comp_feature


def compute_sm_feature(mu, v):
    krange = np.float_(range(0, 72*1))/(1.0)
    krange = np.atleast_2d(krange).T
    grid_num = len(krange)

    kernel_resp_1d = compute_sm_1d(mu, v, krange)
    # add one extra dimensionon to indicate if it is a periodic kernel
    if(mu > np.pi*np.sqrt(v)):
        flag_is_periodic = 10.
    else:
        flag_is_periodic = 0.
    return np.hstack((kernel_resp_1d.flatten(), flag_is_periodic))

