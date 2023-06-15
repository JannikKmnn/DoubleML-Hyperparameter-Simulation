import numpy as np
import doubleml as dml
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from itertools import product
from doubleml import DoubleMLData
#from doubleml._utils import _rmse
from doubleml.datasets import make_plr_CCDDHNR2018, make_pliv_CHS2015, make_irm_data, make_iivm_data
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import toeplitz

import warnings
warnings.filterwarnings('ignore', category = ConvergenceWarning)

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
    
    np.random.seed(1312)
    
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

# compute in- and out-of-sample MSE
def plr_in_and_out_of_sample_mse(dml_plr_obj: dml.DoubleMLPLR, 
                                 ml_l_dml_model, ml_m_dml_model, 
                                 dml_train_data_obj: dml.DoubleMLData, 
                                 dml_test_data_obj: dml.DoubleMLData):

    """
    
    Computes MSE of Model Predictions in Training- and Test-Data.

    Args:
        dml_plr_obj: DoubleML-PLR-Model to observe causal parameter theta.
        ml_l_dml_model: Nuisance function for regression on y.
        ml_m_dml_model: Nuisance function for regression on d.
        dml_train_data_obj: dml data object for training.
        dml_test_data_obj: dml data object for testing.

    Returns:
        in_sample_mse, out_of_sample_mse (float): mean squared errors to detect performance of the model.
    
    """

    X_train = dml_train_data_obj.data.drop(['y', 'd'], axis=1).to_numpy()
    y_train = dml_train_data_obj.data['y'].to_numpy()
    #d_train = dml_train_data_obj.data['d'].to_numpy()

    X_test = dml_test_data_obj.data.drop(['y', 'd'], axis=1).to_numpy()
    y_test = dml_test_data_obj.data['y'].to_numpy()
    #d_test = dml_test_data_obj.data['d'].to_numpy()

    d_pred_train = ml_m_dml_model.predict(X_train)
    d_train = dml_train_data_obj.data[dml_train_data_obj.d_cols].values
    #y_pred_train = d_pred_train*dml_plr_obj.coef[0] + ml_l_dml_model.predict(X_train)
    y_pred_train = d_train*dml_plr_obj.coef[0] + ml_l_dml_model.predict(X_train)
    in_sample_mse = np.mean((y_train - y_pred_train)**2)

    d_pred_test = ml_m_dml_model.predict(X_test)
    d_test = dml_test_data_obj.data[dml_test_data_obj.d_cols].values
    #y_pred_test = d_pred_test*dml_plr_obj.coef[0] + ml_l_dml_model.predict(X_test)
    y_pred_test = d_test*dml_plr_obj.coef[0] + ml_l_dml_model.predict(X_test)
    out_of_sample_mse = np.mean((y_test - y_pred_test)**2)

    return in_sample_mse, out_of_sample_mse

# PLR gradient boosting simulation
def simulate_gb_plr(data, train_test_split: float, ml_l, ml_m, ml_g=None, n_folds=1, score='partialling out', random_seed=1312):

    """
    Fits data and nuisance models as gradient boosting to PLR model.


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

        dml_obj_data_train = dml.DoubleMLData(dml_obj_data.data[:int((1-train_test_split)*n_obs)], 'y', 'd')
        dml_obj_data_test = dml.DoubleMLData(dml_obj_data.data[int((1-train_test_split)*n_obs):], 'y', 'd')

        if n_folds == 1: apply_cross_fitting = False

        if score == 'partialling out':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data_train,
                                        ml_l, ml_m,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        elif score == 'IV-type':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data_train,
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



# PLR lasso simulation
def simulate_lasso_plr(data, train_test_split: float, ml_l, ml_m, ml_g=None, 
                       apply_cross_fitting=True, n_folds=1, score='partialling out', random_seed=1312):

    """
    Fits data and nuisance models as lasso regressions to PLR model.

    Args:
        data: list of repetitions or single data tuple.
        train_test_split (float): percentage of data used for validation.
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
    in_sample_mses = []
    out_of_sample_mses = []

    for i_rep in range(n_rep):
        
        # either the covariates (x), the treatment (d) and outcome (y) variables are provided directly in tuples
        # or the data come in a pandas dataframe and the name of the treatment and outcome columns are given
        if type(data[i_rep]) == tuple:
            (x, y, d) = data[i_rep]
            dml_obj_data = dml.DoubleMLData.from_arrays(x, y, d)
        else:
            dml_obj_data = dml.DoubleMLData(data, 'y', 'd')

        n_obs = dml_obj_data.n_obs

        dml_obj_data_train = dml.DoubleMLData(dml_obj_data.data[:int((1-train_test_split)*n_obs)], 'y', 'd')
        dml_obj_data_test = dml.DoubleMLData(dml_obj_data.data[int((1-train_test_split)*n_obs):], 'y', 'd')

        # cross fitting is only possible if dataset is split into more than one fold
        if n_folds == 1: apply_cross_fitting = False

        # the score decides wether another nuisance function for an instrumental variable is required (IV-type)
        # or partialling out chosen
        if score == 'partialling out':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data_train,
                                        ml_l, ml_m,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        elif score == 'IV-type':
            dml_plr_obj = dml.DoubleMLPLR(dml_obj_data_train,
                                        ml_l, ml_m, ml_g,
                                        n_folds=n_folds,
                                        score=score,
                                        apply_cross_fitting=apply_cross_fitting)
        else:
            raise ValueError(f"'{score}' is not a valid score function.")

        dml_plr_obj.fit(store_models=True)

        if n_folds > 1:

            for fit in range(n_folds):
                if lassocv_flag:
                    lasso_alphas.append((dml_plr_obj.models['ml_l']['d'][0][fit].alpha_,
                                        dml_plr_obj.models['ml_m']['d'][0][fit].alpha_))
                train_first_indexes = dml_plr_obj.smpls[0][fit][0]
                train_data_fit = dml.DoubleMLData(dml_obj_data_train.data.iloc[train_first_indexes],
                                                  'y', 'd')
                in_sample_mse, out_of_sample_mse = plr_in_and_out_of_sample_mse(
                    dml_plr_obj,
                    dml_plr_obj.models['ml_l']['d'][0][fit],
                    dml_plr_obj.models['ml_m']['d'][0][fit],
                    train_data_fit,
                    dml_obj_data_test
                )
                in_sample_mses.append(in_sample_mse)
                out_of_sample_mses.append(out_of_sample_mse)

        else:
            
            in_sample_mse, out_of_sample_mse = plr_in_and_out_of_sample_mse(
                dml_plr_obj,
                dml_plr_obj.models['ml_l']['d'][0][0],
                dml_plr_obj.models['ml_m']['d'][0][0],
                dml_obj_data_train,
                dml_obj_data_test
            )
            in_sample_mses.append(in_sample_mse)
            out_of_sample_mses.append(out_of_sample_mse)

        this_theta = dml_plr_obj.coef[0]
        this_se = dml_plr_obj.se[0]

        theta_scores[i_rep] = this_theta
        se_scores[i_rep] = this_se

        dml_plr_objects.append(dml_plr_obj)

    return theta_scores, se_scores, dml_plr_objects, in_sample_mses, out_of_sample_mses, lasso_alphas

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
