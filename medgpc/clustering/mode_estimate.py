import os
import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate
from ..util.binaryIO import write_double_to_bin
from ..visualization.vizkernel import plot_one_kernel, plot_kde_hist, plot_cluster_scatter_2d


def output_mode_kernel(fold, exp_param, pan_array, hyp_array, mixture_pan, mixture_index,
                       mixture_cluster_num, mixture_cluster_assign, kernclust_alg,
                       plotting_mode, plotting_param):
    kernel_type = exp_param["kernel"]
    output_mode_by_kernel = {'SE': output_mode_SE, 
                             'SM': output_mode_SM,
                             'LMC-SM': output_mode_LMC_SM}
    try:
        kde_mode_hyp = output_mode_by_kernel[kernel_type](
                            fold, exp_param, pan_array, hyp_array, mixture_pan, mixture_index,
                            mixture_cluster_num, mixture_cluster_assign, kernclust_alg,
                            plotting_mode, plotting_param
                            )
    except KeyError:
        print('Error: specified kernel type {} not supported'.format(kernel_type))
        raise NotImplementedError
    except:
        print('Error: Unknown error occurred in output_mode_kernel()')
        exit(1)
    return kde_mode_hyp

def output_mode_SE(fold, exp_param, pan_array, hyp_array, mixture_pan, mixture_index,
                       mixture_cluster_num, mixture_cluster_assign, kernclust_alg,
                       plotting_mode, plotting_param):
    # create directory for outputs
    if(fold != -1):
        sub_dir = 'fold{}'.format(fold)
    else:
        sub_dir = 'all'
    kde_output_dir = os.path.join(exp_param["exp_kernel_dir"], sub_dir)
    if not os.path.exists(kde_output_dir):
        os.makedirs(kde_output_dir)
    
    if(plotting_mode != 0):
        figure_output_dir = os.path.join(exp_param["exp_figure_dir"], sub_dir, kernclust_alg)
        if not os.path.exists(figure_output_dir):
            os.makedirs(figure_output_dir)

    assert(mixture_cluster_num == 1)

    # estimate mode for nuggets/lengthscale/scalefactor
    kde_mode_hyp = np.zeros(hyp_array.shape[1])
    hyp_name = {0: 'nugget', 1: 'lengthscale', 2: 'scalefactor'}
    for i in range(hyp_array.shape[1]):
        all_hyp = np.exp(hyp_array[:, i])
        assert len(all_hyp) == len(pan_array)
        if(hyp_name[i] == 'lengthscale'):
            kde_lc_range = np.linspace(0.01, 1000., 100001)
            dens, _ = compute_kde(all_hyp, kde_lc_range)
            mode = compute_mode(kde_lc_range, dens, weighted=False)
        else:
            dens, _ = compute_kde(all_hyp, all_hyp)
            mode = compute_mode(all_hyp, dens, weighted=False)
        kde_mode_hyp[i] = np.log(mode)
        if(plotting_mode >= 2):
            plot_kde_hist(all_hyp, dens,
                            title='{}: {:.4f}'.format(hyp_name[i], mode), 
                            fig_format='pdf', fig_name='{}'.format(hyp_name[i]),
                            out_dir=figure_output_dir)

    # output for testing
    prefix = 'mode_'
    np.savetxt(os.path.join(kde_output_dir, '{}_{}mixture_num.txt'.format(kernclust_alg, prefix)), [mixture_cluster_num], fmt='%d')
    mode_file_name = os.path.join(kde_output_dir, '{}_{}param.bin'.format(kernclust_alg, prefix))
    write_double_to_bin(mode_file_name, kde_mode_hyp.flatten())
    print('Info: output final mode parameters to file: {}'.format(mode_file_name))
    
    if(plotting_mode >= 1):
        plot_one_kernel(exp_param["kernel"], kde_mode_hyp, 
                        fig_format='pdf', fig_out_dir=figure_output_dir,
                        fig_prefix=prefix)
    return kde_mode_hyp


def output_mode_SM(fold, exp_param, pan_array, hyp_array, mixture_pan, mixture_index,
                    mixture_cluster_num, mixture_cluster_assign, kernclust_alg,
                    plotting_mode, plotting_param):
    # create directory for outputs
    if(fold != -1):
        sub_dir = 'fold{}'.format(fold)
    else:
        sub_dir = 'all'
    kde_output_dir = os.path.join(exp_param["exp_kernel_dir"], sub_dir)
    if not os.path.exists(kde_output_dir):
        os.makedirs(kde_output_dir)

    if(plotting_mode != 0):
        figure_output_dir = os.path.join(exp_param["exp_figure_dir"], sub_dir, kernclust_alg)
        if not os.path.exists(figure_output_dir):
            os.makedirs(figure_output_dir)

    # read parameters
    Q = exp_param["Q"]
    assert(exp_param["D"] == 1)

    newQ = mixture_cluster_num
    kde_param_num = 1 + 3*newQ
    kde_mode_hyp = np.zeros(kde_param_num)

    # estimate mode for nuggets
    all_nu = np.exp(hyp_array[:, 0])
    assert len(all_nu) == len(pan_array)
    
    dens, _ = compute_kde(all_nu, all_nu)
    mode_nu = compute_mode(all_nu, dens, weighted=False)
    
    kde_mode_hyp[0] = np.log(mode_nu)
    if(plotting_mode >= 2):
        plot_kde_hist(all_nu, dens,
                        title='nugget: {:.4f}'.format(mode_nu), 
                        fig_format='pdf', fig_name='nugget',
                        out_dir=figure_output_dir)

    # estimate mode for (1) elements in B and (2) mu & v for each kernel cluster
    cluster_id_array = np.unique(mixture_cluster_assign)
    assert len(cluster_id_array) == mixture_cluster_num

    # visualize all
    all_comp_per = np.zeros(len(mixture_cluster_assign))
    all_comp_lc = np.zeros(len(mixture_cluster_assign))
    for x, c in enumerate(mixture_cluster_assign):
        pan = mixture_pan[x]
        qq = mixture_index[x]
        comp2kern_idx = np.where(pan_array == pan)[0]
        assert len(comp2kern_idx) == 1
        hyp = hyp_array[comp2kern_idx[0], :]
        all_comp_per[x] = 1./np.exp(hyp[1 + Q + qq])
        all_comp_lc[x] = 1./(2.*np.pi*np.exp(hyp[1 + 2*Q + qq]))
    plot_cluster_scatter_2d(x           =   all_comp_per            , \
                            y           =   all_comp_lc             , \
                            cluster     =   mixture_cluster_assign  , \
                            xmin        =   0.0                     , \
                            xmax        =   200.0                   , \
                            ymin        =   0.0                     , \
                            ymax        =   500.0                   , \
                            title       =   '# of cluster: ' + str(mixture_cluster_num), \
                            xlabel      =   'period'                , \
                            ylabel      =   'lengthscale'           , \
                            fig_name    =   'all_cluster_feature'   , \
                            fig_format  =   'pdf'                   , \
                            out_dir     =   figure_output_dir)


    for q, cluster_id in enumerate(cluster_id_array):
        print('Info: estimate new parameters for new cluster {}'.format(q))
        clust2comp_idx = np.where(mixture_cluster_assign == cluster_id)[0]
        assert len(clust2comp_idx) > 0
        print('Info: cluster size {}; ({:2.2f}%)'.format(
            len(clust2comp_idx), 100.*len(clust2comp_idx)/len(mixture_cluster_assign)))

        # estimate new mu & (square root) v 
        all_mu = np.zeros(len(clust2comp_idx))
        all_v_sr = np.zeros(len(clust2comp_idx))
        for i, cidx in enumerate(clust2comp_idx):
            pan = mixture_pan[cidx]
            qq = mixture_index[cidx]
            comp2kern_idx = np.where(pan_array == pan)[0]
            assert len(comp2kern_idx) == 1
            hyp = hyp_array[comp2kern_idx[0], :]
            all_mu[i] = np.exp(hyp[1 + Q + qq])
            all_v_sr[i] = np.exp(hyp[1 + 2*Q + qq])

        kde_per_range = np.linspace(0.01, 1000., 100001)
        kde_mu_range = 1./kde_per_range
        dens, _ = compute_kde(all_mu, kde_mu_range)
        mode_mu = compute_mode(kde_mu_range, dens, weighted=False)
        kde_mode_hyp[1 + newQ + q] = np.log(mode_mu)
        if(plotting_mode >= 2):
            plot_kde_hist(1./all_mu, dens,
                            title='period {}: {:.2f}'.format(q, 1./mode_mu), 
                            fig_format='pdf', fig_name='period{}'.format(q),
                            out_dir=figure_output_dir)

        kde_lc_range = np.linspace(0.01, 1000., 100001)
        kde_vsr_range = 1./(2.*np.pi*kde_lc_range)
        dens, _ = compute_kde(all_v_sr, kde_vsr_range)
        mode_v_sr = compute_mode(kde_vsr_range, dens, weighted=False)
        kde_mode_hyp[1 + 2*newQ + q] = np.log(mode_v_sr)
        if(plotting_mode >= 2):
            plot_kde_hist(1./(2.*np.pi*all_v_sr), dens, 
                            title='lengthscale {}: {:.2f}'.format(q, 1./(2.*np.pi*mode_v_sr)), 
                            fig_format='pdf', fig_name='lengthscale{}'.format(q),
                            out_dir=figure_output_dir)

        # estimate weights:
        # if there are multiple components from the same subject are in the same cluster,
        # those components' weights are added before kde estimation
        clust_comp_pan = mixture_pan[clust2comp_idx]
        clust_comp_index = mixture_index[clust2comp_idx]
        kernel_cr = (100.*len(np.unique(clust_comp_pan)))/len(pan_array)
        print('coverage of kernel cluster {}: {:2.2f}%'.format(q, kernel_cr))

        all_w = []
        for pan in np.unique(clust_comp_pan):
            comp2kern_idx = np.where(pan_array == pan)[0]
            assert len(comp2kern_idx) == 1
            hyp = hyp_array[comp2kern_idx[0], :]

            w = 0.
            indices = np.where(clust_comp_pan == pan)[0]
            #if(len(indices) > 1):
            #    print('Info: cluster {} contains {} components from case {}'.format(q, len(indices), pan))
            for idx in indices:
                qq = clust_comp_index[idx]
                w += np.exp(hyp[1 + qq])
            all_w.append(w)
        print('Info: total # of weights for kde: {}'.format(len(all_w)))
        all_w = np.asarray(all_w)
        print('Info: maximum absolute value for weights: {:6.6f}'.format(np.max(all_w)))

        # estimate the mode value of each element in B
        dens, _ = compute_kde(all_w, all_w)
        mode_w = compute_mode(all_w, dens, weighted=False)
        if(plotting_mode >= 2):
            plot_kde_hist(all_w, dens,
                           title='$w_{}$: {:.4f}'.format(q, mode_w), 
                           fig_format='pdf', fig_name='w{}'.format(q),
                           out_dir=figure_output_dir)
        kde_mode_hyp[1 + q] = np.log(mode_w)

    # output for testing
    prefix = 'mode_'
    np.savetxt(os.path.join(kde_output_dir, '{}_{}mixture_num.txt'.format(kernclust_alg, prefix)), [mixture_cluster_num], fmt='%d')
    mode_file_name = os.path.join(kde_output_dir, '{}_{}param.bin'.format(kernclust_alg, prefix))
    write_double_to_bin(mode_file_name, kde_mode_hyp.flatten())
    print('Info: output final mode parameters to file: {}'.format(mode_file_name))
    
    if(plotting_mode >= 1):
        plot_one_kernel(exp_param["kernel"], kde_mode_hyp, 
                        fig_format='pdf', fig_out_dir=figure_output_dir,
                        fig_prefix=prefix, Q=newQ)
    return kde_mode_hyp


def output_mode_LMC_SM(fold, exp_param, pan_array, hyp_array, mixture_pan, mixture_index,
                       mixture_cluster_num, mixture_cluster_assign, kernclust_alg,
                       plotting_mode, plotting_param):
    # create directory for outputs
    if(fold != -1):
        sub_dir = 'fold{}'.format(fold)
    else:
        sub_dir = 'all'
    kde_output_dir = os.path.join(exp_param["exp_kernel_dir"], sub_dir)
    if not os.path.exists(kde_output_dir):
        os.makedirs(kde_output_dir)

    if(plotting_mode != 0):
        figure_output_dir = os.path.join(exp_param["exp_figure_dir"], sub_dir, kernclust_alg)
        if not os.path.exists(figure_output_dir):
            os.makedirs(figure_output_dir)

    # read parameters
    Q = exp_param["Q"]
    D = exp_param["D"]
    R = exp_param["R"]
    newQ = mixture_cluster_num
    kde_param_num = D + newQ*(D*R + 2 + D)
    kde_mode_hyp = np.zeros(kde_param_num)

    # estimate mode for nuggets
    for d in range(D):
        all_nu = np.exp(hyp_array[:, d])
        assert len(all_nu) == len(pan_array)

        dens, _ = compute_kde(all_nu, all_nu)
        mode_nu = compute_mode(all_nu, dens, weighted=True)
        kde_mode_hyp[d] = np.log(mode_nu)
        if(plotting_mode >= 2):
            plot_kde_hist(all_nu, dens,
                            title='nugget {}: {:.4f}'.format(d, mode_nu), 
                            fig_format='pdf', fig_name='nugget{}'.format(d),
                            out_dir=figure_output_dir)

    # estimate mode for (1) elements in B and (2) mu & v for each kernel cluster
    cluster_id_array = np.unique(mixture_cluster_assign)
    print(cluster_id_array)
    assert len(cluster_id_array) == mixture_cluster_num

    # visualize all
    all_comp_per = np.zeros(len(mixture_cluster_assign))
    all_comp_lc = np.zeros(len(mixture_cluster_assign))
    for x, c in enumerate(mixture_cluster_assign):
        pan = mixture_pan[x]
        qq = mixture_index[x]
        comp2kern_idx = np.where(pan_array == pan)[0]
        assert len(comp2kern_idx) == 1
        hyp = hyp_array[comp2kern_idx[0], :]
        all_comp_per[x] = 1./np.exp(hyp[D + Q*D*R + qq])
        all_comp_lc[x] = 1./(2.*np.pi*np.exp(hyp[D + Q*D*R + Q + qq]))
    plot_cluster_scatter_2d(x           =   all_comp_per            , \
                            y           =   all_comp_lc             , \
                            cluster     =   mixture_cluster_assign  , \
                            xmin        =   0.0                     , \
                            xmax        =   200.0                   , \
                            ymin        =   0.0                     , \
                            ymax        =   500.0                   , \
                            title       =   '# of cluster: ' + str(mixture_cluster_num), \
                            xlabel      =   'period'                , \
                            ylabel      =   'lengthscale'           , \
                            fig_name    =   'all_cluster_feature'   , \
                            fig_format  =   'pdf'                   , \
                            out_dir     =   figure_output_dir)


    for q, cluster_id in enumerate(cluster_id_array):
        print('Info: estimate new parameters for new cluster {}'.format(q))
        clust2comp_idx = np.where(mixture_cluster_assign == cluster_id)[0]
        assert len(clust2comp_idx) > 0
        print('Info: cluster size {}; ({:2.2f}%)'.format(
            len(clust2comp_idx), 100.*len(clust2comp_idx)/len(mixture_cluster_assign)))
        #print(clust2comp_idx)

        # estimate new mu & (square root) v 
        all_mu = np.zeros(len(clust2comp_idx))
        all_v_sr = np.zeros(len(clust2comp_idx))
        for i, cidx in enumerate(clust2comp_idx):
            pan = mixture_pan[cidx]
            qq = mixture_index[cidx]
            comp2kern_idx = np.where(pan_array == pan)[0]
            assert len(comp2kern_idx) == 1
            hyp = hyp_array[comp2kern_idx[0], :]
            all_mu[i] = np.exp(hyp[D + Q*D*R + qq])
            all_v_sr[i] = np.exp(hyp[D + Q*D*R + Q + qq])

        dens, _ = compute_kde(all_mu, all_mu)
        mode_mu = compute_mode(all_mu, dens, weighted=True)

        kde_mode_hyp[D + newQ*D*R + q] = np.log(mode_mu)
        if(plotting_mode >= 2):
            plot_kde_hist(1./all_mu, dens,
                            title='period {}: {:.2f}'.format(q, 1./mode_mu), 
                            fig_format='pdf', fig_name='period{}'.format(q),
                            out_dir=figure_output_dir)

        dens, _ = compute_kde(all_v_sr, all_v_sr)
        mode_v_sr = compute_mode(all_v_sr, dens, weighted=True)

        kde_mode_hyp[D + newQ*(D*R+1) + q] = np.log(mode_v_sr)
        if(plotting_mode >= 2):
            plot_kde_hist(1./(2.*np.pi*all_v_sr), dens, 
                            title='lengthscale {}: {:.2f}'.format(q, 1./(2.*np.pi*mode_v_sr)), 
                            fig_format='pdf', fig_name='lengthscale{}'.format(q),
                            out_dir=figure_output_dir)

        # estimate new B matrices:
        # if there are multiple components from the same subject are in the same cluster,
        # those components' B matrices are added before kde estimation
        clust_comp_pan = mixture_pan[clust2comp_idx]
        clust_comp_index = mixture_index[clust2comp_idx]
        kernel_cr = (100.*len(np.unique(clust_comp_pan)))/len(pan_array)
        print('coverage of kernel cluster {}: {:2.2f}%'.format(q, kernel_cr))

        all_B = []
        for pan in np.unique(clust_comp_pan):
            comp2kern_idx = np.where(pan_array == pan)[0]
            assert len(comp2kern_idx) == 1
            hyp = hyp_array[comp2kern_idx[0], :]

            B = np.zeros((D, D))
            indices = np.where(clust_comp_pan == pan)[0]
            #if(len(indices) > 1):
            #    print('Info: cluster {} contains {} components from case {}'.format(q, len(indices), pan))
            for idx in indices:
                qq = clust_comp_index[idx]
                A = np.zeros((D, R))
                lam = np.zeros((D,))
                for d in range(D):
                    lam[d] = np.exp(hyp[D + Q*(D*R+2) + qq*D + d])
                    for r in range(R):
                        A[d, r] = hyp[D + qq*(D*R) + d*R + r]
                B += (np.dot(A, A.T) + np.diag(lam.flatten()))
            all_B.append(B)
        print('Info: total # of B matrices for kde: {}'.format(len(all_B)))
        all_B = np.asarray(all_B)
        print('Info: shape of aggregated B matrices: {}'.format(all_B.shape))
        print('Info: maximum absolute value for B matrices: {:6.6f}'.format(np.max(all_B)))

        # final kde estimate kde for B matrices
        kde_B = np.zeros((D, D))
        kde_b_range = np.linspace(-50.0, 50.0, 100001)
        if(np.max(all_B) > 50):
            steps = np.round(np.max(all_B)+1)/0.001
            kde_b_range = np.linspace(-np.round(np.max(all_B)+1), np.round(np.max(all_B)+1), int(2*steps+1))
        print('Info: precision of kde b elements: {}'.format(kde_b_range[1]-kde_b_range[0]))
        print('Info: total # of available elements: {}'.format(len(kde_b_range)))

        # estimate the mode value of each element in B
        for d1 in range(D):
            for d2 in range(d1, D):
                data = all_B[:, d1, d2].flatten()
                dens, _ = compute_kde(data, data)
                mode_b = compute_mode(data, dens, weighted=True)
                kde_B[d1, d2] = mode_b
                kde_B[d2, d1] = mode_b
                if(plotting_mode >= 2):
                    plot_kde_hist(data, dens, test_data=kde_b_range, 
                                   title='B{}({}, {}): {:.4f}'.format(q, d1, d2, mode_b), 
                                   fig_format='pdf', fig_name='B{}_{}_{}'.format(q, d1, d2),
                                   out_dir=figure_output_dir)

        sparsity = (100.*len(np.where(np.abs(kde_B) < 0.001)[0]))/len(kde_B.flatten())
        print('Info: sparsity of kde B matrix: {:2.2f}'.format(sparsity))

        # decompose empirical B matrices to A and lambda
        U, S, V = np.linalg.svd(kde_B)
        A_ = (U*np.sqrt(S))[:, 0:R]
        lam_ = np.diag(kde_B - np.dot(A_, A_.T)).copy()
        print('Info: negative lambda # {}'.format(len(np.where(lam_ <= 0.)[0])))
        lam_[np.where(lam_ <= 0.)] = 1e-15
        for d in range(D):
            kde_mode_hyp[D + newQ*(D*R+2) + q*D + d] = np.log(lam_[d])
            for r in range(R):
                kde_mode_hyp[D + q*(D*R) + d*R + r] = A_[d, r]
        #print('Debugging: mode parameters after estimating components:')
        #print(kde_mode_hyp)

    # output for testing
    prefix = 'mode_'
    np.savetxt(os.path.join(kde_output_dir, '{}_{}mixture_num.txt'.format(kernclust_alg, prefix)), [mixture_cluster_num], fmt='%d')
    mode_file_name = os.path.join(kde_output_dir, '{}_{}param.bin'.format(kernclust_alg, prefix))
    write_double_to_bin(mode_file_name, kde_mode_hyp.flatten())
    print('Info: output final mode parameters to file: {}'.format(mode_file_name))
    
    if(plotting_mode >= 1):
        plot_one_kernel(exp_param["kernel"], kde_mode_hyp, 
                        fig_format='pdf', fig_out_dir=figure_output_dir,
                        fig_prefix=prefix, Q=newQ, D=D, R=R)
    return kde_mode_hyp


def compute_kde(data, test_x):
    data = data.flatten()
    test_x = test_x.flatten()
    kde = KDEUnivariate(data)
    kde.fit(kernel="gau", bw="silverman")
    dens = kde.evaluate(test_x)
    return dens, None

def compute_mode(data, density, weighted=True):
    if(weighted):
        return np.nansum(data*density)/np.nansum(density)
    else:
        return data[np.argmax(density)]

