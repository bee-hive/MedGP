import matplotlib
matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.figsize'] = (12, 12)
    font = {'family' : 'sans-serif',
            'sans-serif' : 'Arial',
            'weight' : 'normal'}
    matplotlib.rc('font', **font)
    import pylab as pb
except:
    pass
import seaborn as sns
sns.set_style('white')
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import numpy as np
from .fastkernel import compute_A_matrix, compute_B_matrix, compute_kappa_vector, compute_sm_1d, compute_se_1d

def plot_kde_hist(  data, test_log_lik,
                    test_data   =   None                , \
                    bins        =   100                 , \
                    sample_min  =   None                , \
                    sample_max  =   None                , \
                    xmin        =   None                , \
                    xmax        =   None                , \
                    ymin        =   None                , \
                    ymax        =   None                , \
                    savefig     =   True                , \
                    title       =   ''                  , \
                    xlabel      =   ''                  , \
                    ylabel      =   ''                  , \
                    fig_name    =   'kde_dist'          , \
                    fig_format  =   'pdf'               , \
                    out_dir     =   './'                ):
    fig = plt.figure(figsize=(12, 12))
    sns.distplot(data.flatten())

    if(xmin is not None):
        plt.xlim(xmin=xmin)
    if(xmax is not None):
        plt.xlim(xmax=xmax)

    if(ymin is not None):
        plt.ylim(ymin=ymin)
    if(ymax is not None):
        plt.ylim(ymax=ymax)

    plt.tick_params(labelsize=12)
    plt.tick_params(
                    axis        ='y'    , \
                    which       ='both' , \
                    left        ='on'   , \
                    right       ='off'   , \
                    labelleft   ='on'  \
                    )
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.savefig(os.path.join(out_dir, '{}.{}'.format(fig_name, fig_format)), format=fig_format)
    plt.close()


def plot_cluster_scatter_2d(x, y , \
                            cluster     =   None                    , \
                            cluster_name=   None                    , \
                            xmin        =   None                    , \
                            xmax        =   None                    , \
                            ymin        =   None                    , \
                            ymax        =   None                    , \
                            title       =   ''                      , \
                            xlabel      =   'period'                , \
                            ylabel      =   'lengthscale'           , \
                            fig_name    =   'all_cluster_feature'   , \
                            fig_format  =   'pdf'                   , \
                            out_dir     =   './'                    ):
    fig = plt.figure(0, figsize=(6, 6))
    if(cluster is None):
        cluster = np.zeros((len(x),))
    cluster_num = len(np.unique(cluster))
    global_color = plt.cm.spectral(np.linspace(0, 1, cluster_num))

    for i in np.unique(cluster):
        cluster_label = 'cluster {}'.format(i) if(cluster_name is None) else(cluster_name[i])
        indices = np.where(cluster == i)[0]
        plt.scatter(x[indices], y[indices], 
                    c=global_color[np.int_(cluster[indices].flatten()), :],
                    label=cluster_label, alpha=0.7
                    )
    lgd = plt.legend(fontsize=12)

    # plt.scatter(x, y, c=colors)
    if(xmin is not None):
        plt.xlim(xmin=xmin)
    if(xmax is not None):
        plt.xlim(xmax=xmax)

    if(ymin is not None):
        plt.ylim(ymin=ymin)
    if(ymax is not None):
        plt.ylim(ymax=ymax)

    plt.tick_params(labelsize=12)
    plt.tick_params(
                    axis        ='y'    , \
                    which       ='both' , \
                    left        ='on'   , \
                    right       ='off'  , \
                    labelleft   ='on'   \
                    )
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.title(title, fontsize=14)
    plt.savefig(os.path.join(out_dir, '{}.{}'.format(fig_name, fig_format)), format=fig_format)
    plt.close()


def plot_one_kernel(kernel, hyp, 
                    fig_format='pdf', fig_out_dir='./', fig_prefix='', **kwargs):
    if not os.path.exists(fig_out_dir):
        os.makedirs(fig_out_dir)
    if(kernel == 'LMC-SM'):
        plot_one_LMCSM( hyp=hyp, fig_format=fig_format, fig_out_dir=fig_out_dir, 
                        fig_prefix=fig_prefix, **kwargs)
    elif(kernel == 'SE'):
        plot_one_SE( hyp=hyp, fig_format=fig_format, fig_out_dir=fig_out_dir, 
                        fig_prefix=fig_prefix, **kwargs)
    elif(kernel == 'SM'):
        plot_one_SM( hyp=hyp, fig_format=fig_format, fig_out_dir=fig_out_dir, 
                        fig_prefix=fig_prefix, **kwargs)
    else:
        print('kernel type {} not supported yet!'.format(kernel))
        raise NotImplementedError


def plot_1d_kernel(input_range_array, input_gram_matrix , \
                    ytick       =   False               , \
                    ymin        =   0.0                 , \
                    ymax        =   1.0                 , \
                    title       =   'kernel 1D'         , \
                    xlabel      =   'distance in time'  , \
                    ylabel      =   'covariance'        , \
                    fig_name    =   'kernel_1d'         , \
                    fig_format  =   'pdf'               , \
                    out_dir     =   './'):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    plt.plot(input_range_array, input_gram_matrix, linewidth = 4, color='b')

    plt.xlim(np.min(input_range_array), np.max(input_range_array))
    plt.ylim(ymin=ymin, ymax=ymax)
    plt.tick_params(labelsize=36)
    if(not ytick):
        plt.tick_params(
                    axis        ='y'    , \
                    which       ='both' , \
                    left        ='on'   , \
                    right       ='on'   , \
                    labelleft   ='off'  \
                    )
    else:
        plt.ylabel(ylabel, fontsize=36)
    plt.xlabel(xlabel, fontsize=36)
    plt.title(title, fontsize=20)
    plt.savefig(os.path.join(out_dir, '{}.{}'.format(fig_name, fig_format)), format=fig_format)
    plt.close()


def plot_2d_kernel(input_range_array, input_gram_matrix , \
                    colobar     =   True                , \
                    xtick       =   False               , \
                    ytick       =   False               , \
                    vmin        =   -2.0                , \
                    vmax        =   2.0                 , \
                    title       =   'kernel 2D'         , \
                    xlabel      =   'distance in time'  , \
                    ylabel      =   'distance in time'  , \
                    fig_name    =   'kernel_2d'         , \
                    fig_format  =   'pdf'               , \
                    out_dir     =   './'):
    # plt.clf()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)

    im = plt.imshow(input_gram_matrix, interpolation='nearest', 
                    cmap=plt.cm.RdBu, vmin=vmin, vmax=vmax)
    plt.tick_params(labelsize=36)
    if(not xtick):
        plt.tick_params(
                    axis        ='x'    , \
                    which       ='both' , \
                    bottom      ='on'   , \
                    top         ='on'   , \
                    labelbottom ='off'  \
                    )
    if(not ytick):
        plt.tick_params(
                    axis        ='y'    , \
                    which       ='both' , \
                    left        ='on'   , \
                    right       ='on'   , \
                    labelleft   ='off'  \
                    )
    plt.xlabel(xlabel, fontsize=36)
    plt.ylabel(ylabel, fontsize=36)
    plt.title(title, fontsize=20)

    if(colobar):
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad="5%")
        plt.colorbar(im, cax=cax)
    plt.savefig(os.path.join(out_dir, '{}.{}'.format(fig_name, fig_format)), format=fig_format)
    plt.close()


def plot_one_LMCSM(hyp, fig_format='pdf', fig_out_dir='./', fig_prefix='', **kwargs):
    assert 'Q' in kwargs
    assert 'D' in kwargs
    assert 'R' in kwargs
    Q = kwargs['Q']
    D = kwargs['D']
    R = kwargs['R']
    
    krange = np.float_(range(0, 120*10))/10.
    amin = -1.0
    amax = 1.0
    bmin = -0.2
    bmax = 0.2
    kernel_y_tick = [False for x in range(Q)]
    kernel_y_tick[0] = True
    kernel_x_label = 'distance in time (hour)'
    kernel_y_label = 'covariance'
    if('krange' in kwargs):
        krange = kwargs['krange']
    if('amin' in kwargs):
        amin = kwargs['amin']
    if('amax' in kwargs):
        amax = kwargs['amax']
    if('bmin' in kwargs):
        bmin = kwargs['bmin']
    if('bmax' in kwargs):
        bmax = kwargs['bmax']
    if('kernel_x_label' in kwargs):
        kernel_x_label = kwargs['kernel_x_label']
    if('kernel_y_label' in kwargs):
        kernel_y_label = kwargs['kernel_y_label']
        
    # compute each component
    A_matrix_list = compute_A_matrix(Q, D, R, hyp)
    B_matrix_list = compute_B_matrix(Q, D, R, hyp)
    lam_vector_list = compute_kappa_vector(Q, D, R, hyp)
    
    krange = np.atleast_2d(krange).T
    # sm_vector_list = compute_SM_vector(Q, D, R, hyp, krange)
    
    for q in range(Q):
        # plot A matrix
        plot_2d_kernel(np.arange(D), A_matrix_list[q],
                        vmin        =   amin                ,
                        vmax        =   amax                ,
                        title       =   ''                  ,
                        xlabel      =   ''                  ,
                        ylabel      =   ''                  ,
                        fig_name    =   fig_prefix + 'a_matrix_' + str(q),
                        fig_format  =   fig_format          ,
                        out_dir     =   fig_out_dir)
        # plot lambda matrix
        plot_2d_kernel(np.arange(D), np.diag(lam_vector_list[q]) ,
                        vmin        =   amin                ,
                        vmax        =   amax                ,
                        title       =   ''                  ,
                        xlabel      =   ''                  ,
                        ylabel      =   ''                  ,
                        fig_name    =   fig_prefix + 'lam_matrix_' + str(q),
                        fig_format  =   fig_format          ,
                        out_dir     =   fig_out_dir)
        # plot B matrix
        plot_2d_kernel(np.arange(D), B_matrix_list[q]  ,
                        vmin        =   bmin                ,
                        vmax        =   bmax                ,
                        title       =   ''                  ,
                        xlabel      =   ''                  ,
                        ylabel      =   ''                  ,
                        fig_name    =   fig_prefix + 'b_matrix_' + str(q),
                        fig_format  =   fig_format          ,
                        out_dir     =   fig_out_dir )
        # plot base SM kernels
        mu = np.exp(hyp[D + Q*D*R + q])
        v = np.exp(2*hyp[D + Q*(D*R+1) + q])
        resp = compute_sm_1d(mu, v, krange)
        plot_1d_kernel(krange, resp,
                        ytick       =   kernel_y_tick[q]    ,
                        ymin        =   -1.2                ,
                        ymax        =   1.2                 ,
                        title       =   ''                  ,
                        xlabel      =   kernel_x_label      ,
                        ylabel      =   kernel_y_label      ,
                        fig_name    =   fig_prefix + 'sm_1d_' + str(q)   ,
                        fig_format  =   fig_format          ,
                        out_dir     =   fig_out_dir)
        

def plot_one_SE(hyp, fig_format='pdf', fig_out_dir='./', fig_prefix='', **kwargs):
    krange = np.float_(range(0, 120*10))/10.
    kernel_x_label = 'distance in time (hour)'
    kernel_y_label = 'covariance'

    if('krange' in kwargs):
        krange = kwargs['krange']
    if('kernel_x_label' in kwargs):
        kernel_x_label = kwargs['kernel_x_label']
    if('kernel_y_label' in kwargs):
        kernel_y_label = kwargs['kernel_y_label']
        
    # compute each component
    krange = np.atleast_2d(krange).T
    lc = np.exp(hyp[1])
    s2 = np.exp(hyp[2])
    resp = compute_se_1d(s2, lc, krange)
    plot_1d_kernel(krange, resp,
                    ytick       =   True                ,
                    ymin        =   -1.2                ,
                    ymax        =   1.2                 ,
                    title       =   ''                  ,
                    xlabel      =   kernel_x_label      ,
                    ylabel      =   kernel_y_label      ,
                    fig_name    =   fig_prefix + 'se_1d',
                    fig_format  =   fig_format          ,
                    out_dir     =   fig_out_dir)


def plot_one_SM(hyp, fig_format='pdf', fig_out_dir='./', fig_prefix='', **kwargs):
    assert 'Q' in kwargs
    Q = kwargs['Q']
    
    krange = np.float_(range(0, 120*10))/10.
    kernel_x_label = 'distance in time (hour)'
    kernel_y_label = 'covariance'
    if('krange' in kwargs):
        krange = kwargs['krange']
    if('kernel_x_label' in kwargs):
        kernel_x_label = kwargs['kernel_x_label']
    if('kernel_y_label' in kwargs):
        kernel_y_label = kwargs['kernel_y_label']
            
    krange = np.atleast_2d(krange).T
    all_resp = np.zeros(krange.shape)
    for q in range(Q):
        w = np.exp(hyp[1 + q])
        mu = np.exp(hyp[1 + Q + q])
        v = np.exp(2*hyp[1 + 2*Q + q])
        resp = w*compute_sm_1d(mu, v, krange)
        all_resp += resp
        
    plot_1d_kernel(krange, all_resp,
                    ytick       =   True                ,
                    ymin        =   -1.2                ,
                    ymax        =   1.2                 ,
                    title       =   ''                  ,
                    xlabel      =   kernel_x_label      ,
                    ylabel      =   kernel_y_label      ,
                    fig_name    =   fig_prefix + 'sm_1d',
                    fig_format  =   fig_format          ,
                    out_dir     =   fig_out_dir)

