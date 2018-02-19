import numpy as np
from sklearn import mixture


def run_clustering_top(algorithm, feature, max_cluster_num=None, init_num=10, max_iter_num=2000):
    if(max_cluster_num is None):
        max_cluster_num = 5
        print('Warning: maximum number of clusters not set; use default value {}'.format(max_cluster_num))
        
    algorithm = str(algorithm)
    if(algorithm == 'None'): # not doing clustering
        print('Warning: clustering algorithm is not specified; skip clustering')
        cluster_num = 1
        cluster_assign = np.int_(np.zeros(feature.shape[0],))
    elif(algorithm == 'gmm'):
        cluster_num, cluster_assign = run_sklearn_gmm(feature, max_cluster_num, init_num, max_iter_num)
    else:
        print('Error: not supported algorithm {}'.format(algorithm))
        raise NotImplementedError
    return cluster_num, cluster_assign
    

def run_sklearn_gmm(feature, max_cluster_num, init_num=50, max_iter_num=2000):
    lowest_bic = np.infty
    n_components_range = np.arange(1, max_cluster_num+1)
    cv_types = ['full']
    best_model = None
    best_cluster_num = None
    best_cluster_assign = None

    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a mixture of Gaussians with EM
            gmm = mixture.GaussianMixture(  n_components=n_components,
                                            covariance_type=cv_type,
                                            max_iter=max_iter_num, n_init=init_num)
            gmm.fit(feature)
            curr_bic = gmm.bic(feature)
            print('BIC = {:.6f} for {} clusters'.format(curr_bic, n_components))
            if(curr_bic < lowest_bic):
                lowest_bic = curr_bic
                best_model = gmm
                best_cluster_num = n_components
                best_cluster_assign = best_model.predict(feature)
    print('best cluster number using gmm clustering: {}'.format(best_cluster_num))
    return best_cluster_num, best_cluster_assign

