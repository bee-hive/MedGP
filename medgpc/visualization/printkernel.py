import numpy as np
from .fastkernel import compute_B_matrix


def print_kernel_info(kernel_type, hyp, Q, D, R):
    # TBD to dictionary format
    if(kernel_type == 'SE'):
        print_one_SE_kernel(hyp)
    elif(kernel_type == 'SM'):
        print_one_SM_kernel(hyp, Q)
    elif(kernel_type == 'LMC-SM'):
        print_one_LMC_SM_kernel(hyp, Q, D, R)
    else:
        print('specified kernel type {} not supported'.format(kernel_type))
        raise NotImplementedError

def print_one_SE_kernel(hyp):
    sf2 = np.exp(hyp[2])
    lc = np.exp(hyp[1])
    print('SE kernel: scalefactor={:6.4f},\t lengthscale={:6.4f}'.format(sf2, lc))

def print_one_SM_kernel(hyp, Q):
    print('SM kernel Q={}'.format(Q))
    print('q,\t period,\t lengthscale')
    for q in range(Q):
        sf2 = np.exp(hyp[1 + q])
        mu = np.exp(hyp[1 + Q + q])
        period = 1./mu
        v = np.exp(2*hyp[1 + 2*Q + q])
        lc = 1./(2*np.pi*np.sqrt(v))
        print('{},\t {:6.4f},\t {:6.4f}'.format(q, period, lc))

def print_one_LMC_SM_kernel(hyp, Q, D, R):
    print('LMC-SM kernel Q={}'.format(Q))
    print('q,\t period,\t lengthscale,\t max(Bq),\t min(Bq)')
    B = compute_B_matrix(Q, D, R, hyp)
    for q in range(Q):
        mu = np.exp(hyp[D + Q*D*R + q])
        v = np.exp(2*hyp[D + Q*(D*R+1) + q])
        period = 1./mu
        lc = 1./(2*np.pi*np.sqrt(v))
        print('{},\t {:6.4f},\t {:6.4f},\t {:6.4f},\t {:6.4f}'.format(
                q, period, lc, np.max(B[q]), np.min(B[q])
                ))

