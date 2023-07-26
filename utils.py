import numpy as np
import doubleml as dml
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from itertools import product
from doubleml import DoubleMLData
#from doubleml._utils import _rmse
from doubleml.datasets import make_plr_CCDDHNR2018, make_pliv_CHS2015, make_irm_data, make_iivm_data
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import clone
from scipy.linalg import toeplitz

import warnings
warnings.filterwarnings('ignore', category = ConvergenceWarning)

# Prepare Simulations as Objects
class Data_Generation:

    def __init__(self, 
                 ml_l_hyperparameters: list,
                 ml_m_hyperparameters: list):
        self.ml_l_hyperparameters = ml_l_hyperparameters
        self.ml_m_hyperparameters = ml_m_hyperparameters

class Lasso_Simulation:

    def __init__(self):
        pass

# create dataset
def make_data_list(n_rep: int, n_obs: int, n_vars: int, model_type: str, \
                   dim_z=None, R2_d=None, R2_y=None, alpha=0.5, alpha_x=None, random_seed=1312):

    """
    Creates Dataset based on DoubleML documentation.

    Args:
        n_rep (int): Number of repetitions for data generating process (Number of datasets created).
        o_obs (int): Number of observations (rows) per dataset.
        n_vars (int): Number of covariates.
        model_type (str): Causal model (plr, pliv, irm or iivm).
        dim_z (int): Number of instruments if pliv or iivm model is chosen.
        R2_d (float): Value of R2_d parameter for irm model.
        R2_y (float): Value of R2_y parameter for irm model.
        alpha (float): True value of causal parameter theta.
        alpha_x (float): True value of alpha_x parameter in iivm model.
        random_seed (int): Seed for reproducibility.

    Returns:
        data (list): Generated repetitions.
    """

    np.random.seed(random_seed)

    data = list()

    for _ in range(n_rep):

        if model_type == 'plr':
            (x, y, d) = make_plr_CCDDHNR2018(alpha=alpha, n_obs=n_obs, n_vars=n_vars, return_type='array')
            data_entry = (x, y, d)
        elif model_type == 'pliv':
            (x, y, d, z) = make_pliv_CHS2015(alpha=alpha, n_obs=n_obs, dim_x=n_vars, dim_z=dim_z, return_type='array')
            data_entry = (x, y, d, z)
        elif model_type == 'irm':
            (x, y, d) = make_irm_data(theta=alpha, n_obs=n_obs, dim_x=n_vars, \
                                      R2_d=R2_d, R2_y=R2_y, return_type='array')
            data_entry = (x, y, d)
        elif model_type == 'iivm':
            (x, y, d, z)= make_iivm_data(theta=alpha, n_obs=n_obs, dim_x=n_vars, alpha_x=alpha_x, return_type='array')
            data_entry = (x, y, d, z)

        data.append(data_entry)

    return data

# create BCH2014 Dataset
def make_BCH2014_data_list(n_rep: int, theta = 0.5, n_obs=100, dim_x = 200, rho = 0.5,
                            R2_d = 0.5, R2_y = 0.5, design = '1a', random_seed=1312):
    
    np.random.seed(random_seed)

    data = list()

    for _ in range(n_rep):

        (x, y, d, true_betas, dgp_info) = make_BCH2014_data(theta=theta, n_obs=n_obs, dim_x=dim_x, rho=rho,
                                         R2_d=R2_d, R2_y=R2_y, design=design, random_seed=random_seed)
        data_entry = (x, y, d)
        data.append(data_entry)

    return data
    
        
def make_BCH2014_data(theta = 0.5, n_obs=100, dim_x = 200, rho = 0.5,
                R2_d = 0.5, R2_y = 0.5, design = '1a', random_seed=1312):
    
    np.random.seed(random_seed)
    
    v = np.random.standard_normal(size=[n_obs, ])
    zeta = np.random.standard_normal(size=[n_obs, ])
    cov_mat = toeplitz([np.power(rho, k) for k in range(dim_x)])
    x = np.random.multivariate_normal(np.zeros(dim_x), cov_mat, size=[n_obs, ])

    if design == '1a':
        beta_y = np.concatenate((1/np.arange(1,6), np.zeros(5),
                                1/np.arange(1,6), np.zeros(dim_x - 15)))
        beta_d = np.concatenate((1/np.arange(1,11), np.zeros(dim_x - 10)))
        
    if design == '2a':
        beta_y= np.concatenate((1/np.power(np.arange(1,6),2), np.zeros(5),
                                1/np.power(np.arange(1,6),2), np.zeros(dim_x - 15)))
        beta_d = np.concatenate((1/np.power(np.arange(1,11),2), np.zeros(dim_x - 10)))

    b_y_sigma_b_y = np.dot(np.dot(cov_mat, beta_y), beta_y)
    b_d_sigma_b_d = np.dot(np.dot(cov_mat, beta_d), beta_d)

    c_y = np.sqrt(R2_y/((1-R2_y) * b_y_sigma_b_y))
    c_d = np.sqrt(R2_d/((1-R2_d) * b_d_sigma_b_d))

    d = np.dot(x, np.multiply(beta_d, c_d)) + v
    y = d * theta + np.dot(x, np.multiply(beta_y, c_y)) + zeta
    
    true_betas = {'beta_y': np.multiply(beta_y, c_y), 'beta_d': np.multiply(beta_d, c_d)}

    y_pred_orcl = y - zeta
    d_pred_orcl = d - v
    #orcl_rmse_y = _rmse(y, y_pred_orcl)
    #orcl_rmse_d = _rmse(d, d_pred_orcl)
    
    #orcl_rmse = {'rmse_y': orcl_rmse_y,
    #             'rmse_d': orcl_rmse_d}
    
    dgp_info = {'R2_y': R2_y,
                'R2_d': R2_d,
                'rho': rho,
                'design': design}
    
    #return x, y, d, true_betas, orcl_rmse, dgp_info
    return x, y, d, true_betas, dgp_info

# PLR gradient boosting simulation
def simulate_gb_plr(data, ml_l, ml_m, ml_g=None, n_folds=1, score='partialling out', random_seed=1312):

    """
    Fits data and nuisance models as gradient boosting to PLR model.

    Args:
        TODO

    Returns:
        TODO

    """

    np.random.seed(random_seed)

    apply_cross_fitting = True

    n_rep = len(data)

    theta_scores = np.zeros(shape=(n_rep,))
    se_scores = np.zeros(shape=(n_rep,))

    gb_regression_flag = False

    if isinstance(ml_l, GradientBoostingRegressor) and isinstance(ml_m, GradientBoostingRegressor):
        gb_regression_flag = True

    dml_plr_objects = []

    for i_rep in range(n_rep):

        if type(data[i_rep]) == tuple:
            (x, y, d) = data[i_rep]
            dml_obj_data = dml.DoubleMLData.from_arrays(x, y, d)
        else:
            dml_obj_data = dml.DoubleMLData(data, 'y', 'd')

        n_obs = dml_obj_data.n_obs

        dml_obj_data = dml.DoubleMLData(dml_obj_data.data, 'y', 'd')

        if n_folds == 1: apply_cross_fitting = False

        if score == 'partialling out':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data,
                                        ml_l, ml_m,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        elif score == 'IV-type':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data,
                                        ml_l, ml_m, ml_g,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        else:
            raise ValueError(f"'{score}' is not a valid score function.")
        
        dml_plr_obj.fit(store_models=True)

        this_theta = dml_plr_obj.coef[0]
        this_se = dml_plr_obj.se[0]

        theta_scores[i_rep] = this_theta
        se_scores[i_rep] = this_se

        dml_plr_objects.append(dml_plr_obj)

    return theta_scores, se_scores, dml_plr_objects

def simulate_gb_irm(data, ml_l, ml_m, n_folds=5, score='ATE', random_seed=1312):

    """
    Fits data and nuisance models as gradient boosting to IRM model. 
    'ml_m' is required to be a classification estimator since the treatment variable $D$ is binary.

    Args:
        TODO

    Returns:
        TODO

    """

    np.random.seed(random_seed)

    apply_cross_fitting = True

    n_rep = len(data)

    theta_scores = np.zeros(shape=(n_rep,))
    se_scores = np.zeros(shape=(n_rep,))

    assert isinstance(ml_m, GradientBoostingClassifier), "'ml_m' needs to be a GradientBoostingClassifier."

    dml_plr_objects = []

    for i_rep in range(n_rep):

        if type(data[i_rep]) == tuple:
            (x, y, d) = data[i_rep]
            dml_obj_data = dml.DoubleMLData.from_arrays(x, y, d)
        else:
            dml_obj_data = dml.DoubleMLData(data, 'y', 'd')

        n_obs = dml_obj_data.n_obs

        dml_obj_data = dml.DoubleMLData(dml_obj_data.data, 'y', 'd')

        if n_folds == 1: apply_cross_fitting = False

        dml_plr_obj = dml.DoubleMLIRM(dml_obj_data,
                                      ml_l, ml_m,
                                      n_folds=n_folds,
                                      score=score,
                                      apply_cross_fitting=apply_cross_fitting)
        
        dml_plr_obj.fit(store_models=True)

        this_theta = dml_plr_obj.coef[0]
        this_se = dml_plr_obj.se[0]

        theta_scores[i_rep] = this_theta
        se_scores[i_rep] = this_se

        dml_plr_objects.append(dml_plr_obj)

    return theta_scores, se_scores, dml_plr_objects



# PLR lasso simulation
def simulate_lasso_plr(data, ml_l, ml_m, ml_g=None, 
                       apply_cross_fitting=True, n_folds=1, score='partialling out', random_seed=1312):

    """
    Fits data and nuisance models as lasso regressions to PLR model.

    Args:
        data: list of repetitions or single data tuple.
        ml_l: g(X) for regression on Y.
        ml_m: m(X) for regression on D.
        ml_g: g(X) optional if IV-type score is chosen.
        apply_cross_fitting (bool): indicator wether cross fitting should be applied or not.
        n_folds (int): Number of folds for cross fitting. If 1, no cross fitting is applied.
        score: 'partialling out' or 'IV-type', different score functions for PLR model.
        random_seed (int): Seed for reproducibility.

    Returns:
        theta_scores (list): Simulated causal parameters.
        se_scores (list): Standard errors in simulation.
    """

    np.random.seed(random_seed)

    n_rep = len(data)

    theta_scores = np.zeros(shape=(n_rep,))
    se_scores = np.zeros(shape=(n_rep,))

    # alphas for cross-validated lasso regression
    lassocv_flag = False

    if isinstance(ml_l, LassoCV) and isinstance(ml_m, LassoCV):
        lassocv_flag = True

    lasso_alphas = []

    dml_plr_objects = []

    for i_rep in range(n_rep):
        
        # either the covariates (x), the treatment (d) and outcome (y) variables are provided directly in tuples
        # or the data come in a pandas dataframe and the name of the treatment and outcome columns are given
        if type(data[i_rep]) == tuple:
            (x, y, d) = data[i_rep]
            dml_obj_data = dml.DoubleMLData.from_arrays(x, y, d)
        else:
            dml_obj_data = dml.DoubleMLData(data, 'y', 'd')

        n_obs = dml_obj_data.n_obs

        dml_obj_data = dml.DoubleMLData(dml_obj_data.data, 'y', 'd')

        # cross fitting is only possible if dataset is split into more than one fold
        if n_folds == 1: apply_cross_fitting = False

        # the score decides wether another nuisance function for an instrumental variable is required (IV-type)
        # or partialling out chosen
        if score == 'partialling out':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data,
                                        ml_l, ml_m,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        elif score == 'IV-type':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data,
                                        ml_l, ml_m, ml_g,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        else:
            raise ValueError(f"'{score}' is not a valid score function.")

        dml_plr_obj.fit(store_models=True)

        if lassocv_flag:
            for fit in range(n_folds):
                lasso_alphas.append((dml_plr_obj.models['ml_l']['d'][0][fit].alpha_,
                                     dml_plr_obj.models['ml_m']['d'][0][fit].alpha_))

        this_theta = dml_plr_obj.coef[0]
        this_se = dml_plr_obj.se[0]

        theta_scores[i_rep] = this_theta
        se_scores[i_rep] = this_se

        dml_plr_objects.append(dml_plr_obj)

    return theta_scores, se_scores, dml_plr_objects, lasso_alphas

# plot theta distribution from Lasso Regression
def plot_lasso_score(ml_l, ml_m, theta_scores: list, se_scores: list, alpha: float):

    """
    Plots distribution of simulated thetas and standard normal distribution.

    Args:
        ml_l, ml_m: Nuisance functions in PLR model.
        theta_scores (list): Simulated thetas for distribution plotting.
        se_scores (list): Simulated standard errors for distribution plotting.
        alpha (float): True value of theta for normalization.

    Returns:
        Plots scores with matplotlib.
    """
    
    face_colors = sns.color_palette('pastel')
    edge_colors = sns.color_palette('dark')

    plt.figure(constrained_layout=False)

    if ml_l.__class__ == Lasso:
        plt.title(f'Lasso Regression: \n' + '$\\alpha_{m_{0}(x)}$' + f'={ml_m.alpha}'
                '\n' + '$\\alpha_{g_{0}(x)}$' + f'={ml_l.alpha}')
        label = "Double ML Lasso"
    elif ml_l.__class__ == LassoCV:
        plt.title(f'Lasso Regression (Cross Validation): \n' + '$\\alpha_{m_{0}(x)}$' + f'={ml_m.alphas}'
                '\n' + '$\\alpha_{g_{0}(x)}$' + f'={ml_l.alphas}')
        label = "Double ML LassoCV"
                
    ax = sns.histplot((theta_scores - alpha)/se_scores,
                    color=face_colors[2], edgecolor = edge_colors[2],
                    stat='density', bins=30, label=label)
    ax.axvline(0., color='k')

    xx = np.arange(-5, +5, 0.001)
    yy = stats.norm.pdf(xx)

    ax.plot(xx, yy, color='k', label='$\\mathcal{N}(0, 1)$')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.set_xlim([-6., 6.])
    ax.set_xlabel('$(\hat{\\theta}_0 - \\theta_0)/\hat{\sigma}$')

    plt.tight_layout()
    plt.show()

# plot results

def plot_lasso_variation_results(ml_l_hyperparameters, ml_m_hyperparameters,
                                 n_folds, data, true_alpha,
                                 face_colors = sns.color_palette('summer_r'),
                                 edge_colors = sns.color_palette('dark'),
                                 title="Hyperparameter Variation for Lasso Regression",
                                 save_figure=True,
                                 filename=""):
    """
    Simulates different hyperparameter combinations, plots the estimated distributions and returns absolute bias and coverage.

    Args:
        TODO

    Returns:
        TODO
    """

    xx = np.arange(-5, +5, 0.001)
    yy = stats.norm.pdf(xx)

    coverage_scores = dict()
    bias_scores = dict()

    fig, axs = plt.subplots(len(ml_l_hyperparameters), len(ml_m_hyperparameters), 
                            figsize=(10*len(ml_l_hyperparameters), 10*len(ml_m_hyperparameters)), 
                            constrained_layout=True)

    fig.suptitle(title, fontsize=50)

    for i_ml_m in ml_m_hyperparameters:
        for i_ml_l in ml_l_hyperparameters:

            i_m = ml_m_hyperparameters.index(i_ml_m)
            i_l = ml_l_hyperparameters.index(i_ml_l)

            ml_l = Lasso(alpha=i_ml_l)
            ml_m = Lasso(alpha=i_ml_m)
            
            theta_scores, se_scores, plr_objects, _ = simulate_lasso_plr(ml_l=ml_l, ml_m=ml_m, 
                                                        n_folds=n_folds, data=data,
                                                        score='partialling out')
            
            coverage_score = coverage(true_alpha, plr_objects)
            coverage_scores[(i_ml_l, i_ml_m)] = coverage_score

            absolute_bias = abs_bias(true_alpha, theta_scores)
            bias_scores[(i_ml_l, i_ml_m)] = absolute_bias

            axs[i_l, i_m].hist((theta_scores - true_alpha)/se_scores,
                                color=face_colors[2], edgecolor = edge_colors[2],
                                density=True, bins=30, label='Double ML Lasso')
            axs[i_l, i_m].axvline(0., color='k')
            axs[i_l, i_m].plot(xx, yy, color='k', label='$\\mathcal{N}(0, 1)$')
            #axs[i_l, i_m].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            axs[len(ml_l_hyperparameters)-1, i_m].set_xlabel('$\\lambda_{m_{0}(x)}$=' + f'{i_ml_m}', fontsize=35)
            axs[i_l, 0].set_ylabel('$\\lambda_{g_{0}(x)}$=' + f'{i_ml_l}', fontsize=35)
            axs[i_l, i_m].set_xlim([-6., 6.])

    if save_figure: 
        plt.savefig(f"plots/{filename}", facecolor="white")

    return coverage_scores, bias_scores

def plot_lasso_abs_bias(ml_l_hyperparameters, ml_m_hyperparameters, bias_scores,
                        title="Hyperparameter Combinations for Lasso Regression: Absolute Bias",
                        save_fig=True, filename=""):

    """
    TODO write function documentation
    """
    
    bias_list = [(k[0], k[1], v) for k, v in bias_scores.items()]
    lasso_bias_df = pd.DataFrame(bias_list, columns=['ml_l_alphas', 'ml_m_alphas', 'bias'])
    pivot_table_bias = lasso_bias_df.pivot(index='ml_l_alphas', columns='ml_m_alphas', values='bias')

    plt.figure(figsize=(2.5*len(ml_l_hyperparameters), 2*len(ml_m_hyperparameters)))
    plt.title(f"{title}", fontsize=14)
    sns.heatmap(pivot_table_bias, cmap='summer', annot=True)
    plt.xlabel('$\\lambda_{m_{0}(x)}$', fontsize=14)
    plt.ylabel('$\\lambda_{g_{0}(x)}$', fontsize=14)

    if save_fig:
        plt.savefig(f"plots/{filename}", facecolor="white")

    plt.show()

def plot_lasso_coverage(ml_l_hyperparameters, ml_m_hyperparameters, coverage_scores,
                        title="Hyperparameter Combinations for Lasso Regression: Coverage (%)",
                        save_fig=True, filename=""):
    
    """
    TODO write function documentation
    """

    coverage_list = [(k[0], k[1], v*100) for k, v in coverage_scores.items()]
    lasso_coverage_df = pd.DataFrame(coverage_list, columns=['ml_l_alphas', 'ml_m_alphas', 'coverage'])
    pivot_table_coverage = lasso_coverage_df.pivot(index='ml_l_alphas', columns='ml_m_alphas', values='coverage')

    plt.figure(figsize=(2.5*len(ml_l_hyperparameters), 2*len(ml_m_hyperparameters)))
    plt.title(f"{title}", fontsize=14)
    sns.heatmap(pivot_table_coverage, cmap='summer_r', annot=True, fmt='.2f')
    plt.xlabel('$\\lambda_{m_{0}(x)}$', fontsize=14)
    plt.ylabel('$\\lambda_{g_{0}(x)}$', fontsize=14)

    if save_fig:
        plt.savefig(f"plots/{filename}", facecolor="white")

    plt.show()

def plot_bias_coverage_next_to_eachother(ml_l_hyperparameters, ml_m_hyperparameters,
                                         bias_scores, coverage_scores,
                                         suptitle="Hyperparameter Combinations for Lasso Regression",
                                         bias_title="Mean Absolute Bias",
                                         coverage_title="Coverage (%)",
                                         xlabel='$\\lambda_{m_{0}(x)}$',
                                         ylabel='$\\lambda_{g_{0}(x)}$',
                                         save_fig=True, filename=""):

    """
    TODO write function documentations
    """

    bias_list = [(k[0], k[1], v) for k, v in bias_scores.items()]
    lasso_bias_df = pd.DataFrame(bias_list, columns=['ml_l_alphas', 'ml_m_alphas', 'bias'])
    pivot_table_bias = lasso_bias_df.pivot(index='ml_l_alphas', columns='ml_m_alphas', values='bias')

    coverage_list = [(k[0], k[1], v*100) for k, v in coverage_scores.items()]
    lasso_coverage_df = pd.DataFrame(coverage_list, columns=['ml_l_alphas', 'ml_m_alphas', 'coverage'])
    pivot_table_coverage = lasso_coverage_df.pivot(index='ml_l_alphas', columns='ml_m_alphas', values='coverage')

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(5*len(ml_l_hyperparameters), 2*len(ml_m_hyperparameters)))
    fig.suptitle(suptitle, fontsize=16)
    
    g1 = sns.heatmap(pivot_table_bias, cmap='summer', annot=True, ax=ax1)
    g1.set_title(f"{bias_title}", fontsize=14)
    g1.set_xlabel(xlabel, fontsize=12)
    g1.set_ylabel(ylabel, fontsize=12)

    g2 = sns.heatmap(pivot_table_coverage, cmap='summer_r', annot=True, fmt='.2f', ax=ax2)
    g2.set_title(f"{coverage_title}", fontsize=14)
    g2.set_xlabel(xlabel, fontsize=12)

    fig.tight_layout()

    if save_fig:
        plt.savefig(f"plots/{filename}", facecolor="white")

    plt.show()
    

# Gradient Boosting Results
def plot_gb_plr_variation_results(ml_l_hyperparameters, ml_m_hyperparameters,
                                 n_folds, data, true_alpha, tunable_hyperparameter: str,
                                 ml_l_model, ml_m_model,
                                 face_colors = sns.color_palette('summer_r'),
                                 edge_colors = sns.color_palette('dark'),
                                 title="Hyperparameter Variation for Gradient Boosting",
                                 xlabels='${m_{0}(x)}$=', ylabel='${g_{0}(x)}$=',
                                 save_figure=True,
                                 filename=""):
    
    """
    TODO write function documentation
    """

    xx = np.arange(-5, +5, 0.001)
    yy = stats.norm.pdf(xx)

    coverage_scores = dict()
    bias_scores = dict()

    #scores = []
    distributions_calculated = 1

    fig, axs = plt.subplots(len(ml_l_hyperparameters), len(ml_m_hyperparameters), 
                            figsize=(10*len(ml_l_hyperparameters), 10*len(ml_m_hyperparameters)), 
                            constrained_layout=True)

    fig.suptitle(f"{title}", fontsize=40)

    for ml_l_param in ml_l_hyperparameters:
        for ml_m_param in ml_m_hyperparameters:

            i_m = ml_m_hyperparameters.index(ml_m_param)
            i_l = ml_l_hyperparameters.index(ml_l_param)

            ml_l = clone(ml_l_model).set_params(**{tunable_hyperparameter: ml_l_param})
            ml_m = clone(ml_m_model).set_params(**{tunable_hyperparameter: ml_m_param})
            
            theta_scores, se_scores, model_objects = simulate_gb_plr(ml_l=ml_l, ml_m=ml_m, n_folds=n_folds, data=data, score='partialling out')

            coverage_score = coverage(true_alpha, model_objects)
            coverage_scores[(ml_l_param, ml_m_param)] = coverage_score

            absolute_bias = abs_bias(true_alpha, theta_scores)
            bias_scores[(ml_l_param, ml_m_param)] = absolute_bias

            #scores.append((theta_scores, se_scores, model_objects,
            #               f"ml_l-{tunable_hyperparameter}: {ml_l_param}", 
            #               f"ml_m-{tunable_hyperparameter}: {ml_m_param}"))

            print(f"Distributions calculated: {distributions_calculated}")
            
            axs[i_l, i_m].hist((theta_scores - true_alpha)/se_scores,
                                color=face_colors[2], edgecolor = edge_colors[2],
                                density=True, bins=30, label='Double ML Gradient Boosting')
            axs[i_l, i_m].axvline(0., color='k')
            axs[i_l, i_m].plot(xx, yy, color='k', label='$\\mathcal{N}(0, 1)$')
            #axs[i_l, i_m].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            axs[len(ml_l_hyperparameters)-1, i_m].set_xlabel(xlabels + f'{ml_m_param}', fontsize=20)
            axs[i_l, 0].set_ylabel(ylabel + f'{ml_l_param}', fontsize=30)
            axs[i_l, i_m].set_xlim([-6., 6.])

            distributions_calculated += 1

    if save_figure:
        plt.savefig(f"plots/{filename}", facecolor="white")

    plt.show()

    return coverage_scores, bias_scores#, scores

def plot_gb_irm_variation_results(ml_l_hyperparameters, ml_m_hyperparameters,
                                 n_folds, data, true_alpha, tunable_hyperparameter: str,
                                 ml_l_model, ml_m_model,
                                 face_colors = sns.color_palette('summer_r'),
                                 edge_colors = sns.color_palette('dark'),
                                 title="Hyperparameter Variation for Gradient Boosting",
                                 xlabels='${m_{0}(x)}$=', ylabel='${g_{0}(x)}$=',
                                 save_figure=True,
                                 filename=""):
    
    """
    TODO write function documentation
    """

    xx = np.arange(-5, +5, 0.001)
    yy = stats.norm.pdf(xx)

    coverage_scores = dict()
    bias_scores = dict()

    #scores = []
    distributions_calculated = 1

    fig, axs = plt.subplots(len(ml_l_hyperparameters), len(ml_m_hyperparameters), 
                            figsize=(10*len(ml_l_hyperparameters), 10*len(ml_m_hyperparameters)), 
                            constrained_layout=True)

    fig.suptitle(f"{title}", fontsize=40)

    for ml_l_param in ml_l_hyperparameters:
        for ml_m_param in ml_m_hyperparameters:

            i_m = ml_m_hyperparameters.index(ml_m_param)
            i_l = ml_l_hyperparameters.index(ml_l_param)

            ml_l = clone(ml_l_model).set_params(**{tunable_hyperparameter: ml_l_param})
            ml_m = clone(ml_m_model).set_params(**{tunable_hyperparameter: ml_m_param})
            
            theta_scores, se_scores, model_objects = simulate_gb_irm(ml_l=ml_l, ml_m=ml_m, n_folds=n_folds, data=data, score='ATE')

            coverage_score = coverage(true_alpha, model_objects)
            coverage_scores[(ml_l_param, ml_m_param)] = coverage_score

            absolute_bias = abs_bias(true_alpha, theta_scores)
            bias_scores[(ml_l_param, ml_m_param)] = absolute_bias

            #scores.append((theta_scores, se_scores, model_objects,
            #               f"ml_l-{tunable_hyperparameter}: {ml_l_param}", 
            #               f"ml_m-{tunable_hyperparameter}: {ml_m_param}"))

            print(f"Distributions calculated: {distributions_calculated}")
            
            axs[i_l, i_m].hist((theta_scores - true_alpha)/se_scores,
                                color=face_colors[2], edgecolor = edge_colors[2],
                                density=True, bins=30, label='Double ML Gradient Boosting')
            axs[i_l, i_m].axvline(0., color='k')
            axs[i_l, i_m].plot(xx, yy, color='k', label='$\\mathcal{N}(0, 1)$')
            #axs[i_l, i_m].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            axs[len(ml_l_hyperparameters)-1, i_m].set_xlabel(xlabels + f'{ml_m_param}', fontsize=20)
            axs[i_l, 0].set_ylabel(ylabel + f'{ml_l_param}', fontsize=30)
            axs[i_l, i_m].set_xlim([-6., 6.])

            distributions_calculated += 1

    if save_figure:
        plt.savefig(f"plots/{filename}", facecolor="white")

    plt.show()

    return coverage_scores, bias_scores#, scores


# TODO write cv results function

def plot_cv_combinations():
    """
    
    """
    pass

#coverage calculation
def cover_true(theta, confint):

    """

    Function to check whether theta is contained in confindence interval.
    Returns 1 if true and 0 otherwise.
    
    """

    covers_theta = (confint[0] < theta and theta < confint[1])
    
    if covers_theta:
        return 1
    else:
        return 0
    
def coverage(theta, dml_objects):

    """

    Calculates the percentage of thetas in confidence intervals across a list of fitted dml objects.

    TODO
    """

    num_coverages = sum([cover_true(theta, obj.confint().loc['d']) for obj in dml_objects])
    coverage_ = num_coverages/len(dml_objects)

    return coverage_

# calculation of absolute bias

def abs_bias(theta_true, est_thetas):

    """

    Calculates absolute bias between true and estimated causal parameter.

    TODO
    """

    abs_bias = np.abs(est_thetas - theta_true).mean()
    return abs_bias
    