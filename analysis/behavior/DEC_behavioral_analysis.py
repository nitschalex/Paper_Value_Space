#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# BEHAVIORAL ANALYSIS OF DECISION MAKING TASK
#
# AUTHOR: Alexander Nitsch
# CONTACT: nitsch@cbs.mpg.de
# Max Planck Institute for Human Cognitive and Brain Sciences
# DATE: 2022
#
# =============================================================================
"""

#%% IMPORT PACKAGES

from valspa.project_info import *
from statsmodels.stats.anova import AnovaRM


#%% SET VARIABLES

task = 'DEC'

# subjects to run analysis on:
# INITIALLY, run analysis on all subjects to check subject inclusion based on performance
# AT A LATER STAGE, run analysis only on included subjects
analysis_on_all_subs = False
if analysis_on_all_subs:
    subs = copy.deepcopy(valspa.bids_subs) 
else:
    subs = copy.deepcopy(valspa.subs) 
    
# general dataframe aggregations to apply for most analyses
aggregations = {
    'score': ['mean','std','sem'],
    'rt': ['mean','std','sem','min','max'],
    'rt_log': ['mean','std','sem','min','max']    
    }


#%% IMPORT LOGFILES AND ADD ADDITIONAL VARIABLES

# Import subject-specific logfiles and later concatenate all into one df 

# dict with subject-specifc dfs
data_per_sub = {}
data_per_sub_all_trials = {}


for sub in subs:

    """
    ______________
    IMPORT LOGFILE
    """
    
    log_file = glob.glob(valspa.behav_data_path + '/' + sub + '_' + valspa.dec.logfile_suffix + '*' + '.csv')[0]
    
    # data of all trials including passive trials
    data_per_sub_all_trials[sub] = pd.read_csv(log_file, dtype={'subject': str})
    
    # only active trials
    data_per_sub[sub]   = data_per_sub_all_trials[sub][data_per_sub_all_trials[sub]['active_trial']==1]
    data_per_sub[sub]   = data_per_sub[sub].reset_index(drop=True)


    """
    _______________________
    ADD ADDITIONAL VARIABLES
    """
    
    # indicate whether options at active timepoint have the same values (this is the case
    # if the correct reponse for active timepoint indicates 2) and set scores and rts to NaNs
    data_per_sub[sub]['same_value'] = data_per_sub[sub].apply(lambda x: (x[f'timepoint_{x.active_timepoint}_corr_resp']==2), axis=1)
    data_per_sub[sub]['same_value'] = data_per_sub[sub]['same_value'].astype(int)
    data_per_sub[sub].loc[data_per_sub[sub]['same_value']==1, 'score'] = np.nan
    data_per_sub[sub].loc[data_per_sub[sub]['same_value']==1, 'rt'] = np.nan
    
    # label TPs in switch trajectories as pre, switch, post
    data_per_sub[sub]['switch_pre_post'] = np.nan
    data_per_sub[sub].loc[data_per_sub[sub]['active_timepoint']==data_per_sub[sub]['switch_point'], 'switch_pre_post'] = 'switch'
    data_per_sub[sub].loc[data_per_sub[sub]['active_timepoint']==data_per_sub[sub]['switch_point']-1, 'switch_pre_post'] = 'pre'
    data_per_sub[sub].loc[data_per_sub[sub]['active_timepoint']==data_per_sub[sub]['switch_point']+1, 'switch_pre_post'] = 'post'
    
    # log-transform reaction time
    data_per_sub[sub]['rt_log'] = np.log(data_per_sub[sub]['rt'])
    
    # distance of choice location to 45°diagonal 
    data_per_sub[sub]['dist_to_diagonal'] = data_per_sub[sub].apply(lambda x: 
                                                                    abs(np.cross(valspa.dec.diag_p2-valspa.dec.diag_p1, [x[f'timepoint_{x.active_timepoint}_val_A'], x[f'timepoint_{x.active_timepoint}_val_B']]-valspa.dec.diag_p1) / np.linalg.norm(valspa.dec.diag_p2-valspa.dec.diag_p1)),
                                                                    axis=1)
    
    # quandrant of angle (gain / loss / mixed trajectory)
    # cardinal directions coded as 0
    data_per_sub[sub]['angle_quadrant'] = 0
    data_per_sub[sub].loc[data_per_sub[sub]['angle'].isin(range(10,90,10)), 'angle_quadrant'] = 1
    data_per_sub[sub].loc[data_per_sub[sub]['angle'].isin(range(100,180,10)), 'angle_quadrant'] = 2
    data_per_sub[sub].loc[data_per_sub[sub]['angle'].isin(range(190,270,10)), 'angle_quadrant'] = 3
    data_per_sub[sub].loc[data_per_sub[sub]['angle'].isin(range(280,360,10)), 'angle_quadrant'] = 4
    
    # angle parallel or perpendicular to the 45°-diagonal 
    data_per_sub[sub]['angle_ref_diagonal'] = 'other'
    data_per_sub[sub].loc[data_per_sub[sub]['angle'].isin([40, 50, 220, 230]), 'angle_ref_diagonal'] = 'parallel'
    data_per_sub[sub].loc[data_per_sub[sub]['angle'].isin([130, 140, 310, 320]), 'angle_ref_diagonal'] = 'perpendicular'  
    
    # labels for the correct and the chosen option as [1,2]
    data_per_sub[sub]['correct_option'] = data_per_sub[sub].apply(lambda x: 1 if x[f'timepoint_{x.active_timepoint}_val_A'] > x[f'timepoint_{x.active_timepoint}_val_B']
                                                                  else 2 if x[f'timepoint_{x.active_timepoint}_val_A'] < x[f'timepoint_{x.active_timepoint}_val_B']
                                                                  else np.nan,
                                                                  axis=1) 
    
    data_per_sub[sub]['chosen_option'] = data_per_sub[sub].apply(lambda x: 1 if x['given_resp']==0 and x[f'timepoint_{x.active_timepoint}_pos_A']==0
                                                                 else 1 if x['given_resp']==1 and x[f'timepoint_{x.active_timepoint}_pos_A']==1
                                                                 else 2 if x['given_resp']==1 and x[f'timepoint_{x.active_timepoint}_pos_A']==0
                                                                 else 2 if x['given_resp']==0 and x[f'timepoint_{x.active_timepoint}_pos_A']==1
                                                                 else np.nan,
                                                                 axis=1)
    data_per_sub[sub].loc[data_per_sub[sub]['same_value']==1, 'chosen_option'] = np.nan
    
    

# concatenate all into across-subject df
data = pd.concat(data_per_sub[sub] for sub in data_per_sub)
# data = data.apply(pd.to_numeric, errors='ignore') # convert df to type numeric and ignore columns that can't be converted


#%% SUBJECT INCLUSION BASED ON OVERALL PERFORMANCE AND NaNs FOR MISSING RESPONSES

# Include subjects in subsequent analyses if overall performance > 70%
# For inclusion criterion only: keep missing responses included with score=0 

if analysis_on_all_subs:
    data_overall = valspa.aggregate_data(data, ['subject'], aggregations.copy())
    
    perf_crit     = 0.7
    subs_included = list(set(list(data_overall.subject)) - set(list(data_overall[(data_overall['score_mean']<perf_crit)].subject)))
    subs_included.sort()
    
    # save to pkl
    with open(os.path.join(valspa.pt_dataset_path, 'subject_inclusion.pkl'), 'rb') as f:
        pkl_sub_inclusion = pickle.load(f)
        
    pkl_sub_inclusion.update({'behav_subs_included': subs_included})
    
    with open(os.path.join(valspa.pt_dataset_path, 'subject_inclusion.pkl'), 'wb') as f:
        pickle.dump(pkl_sub_inclusion, f)


# NaNs for missing responses
# (over all subjects, 31 trials had missing responses, were randomly distributed,
# no systematic patterns or particular subjects) 
# data_miss_resp = data[data.given_resp.isnull()]
for sub in data_per_sub:
    data_per_sub[sub].loc[data_per_sub[sub]['given_resp'].isnull(),'score'] = np.nan
# concatenate all subject-specific dfs into a single df with all subjects
data = pd.concat(data_per_sub[sub] for sub in data_per_sub)
data = data.reset_index(drop=True)
# data = data.apply(pd.to_numeric, errors='ignore') # convert df to type numeric and ignore columns that can't be converted


data_overall = valspa.aggregate_data(data, ['subject'], aggregations.copy())
data_overall['measure'] = 'score'
data_overall.score_mean.mean()
data_overall.score_mean.std()


#%% PERFORMANCE AND REACTION TIME BASED ON SPECIFIED CONDTIONS

# Aggregate based on conditions here, plot figures for paper later.

# switch vs. non-switch trajactories
data_sw_nsw, data_sw_nsw_group = valspa.aggregate_data(data, ['subject', 'switch'], aggregations.copy())

# TP in switch trajectories: pre, switch, post
data_sw_pre_post, data_sw_pre_post_group = valspa.aggregate_data(data, ['subject', 'switch_pre_post'], aggregations.copy())

# Short- vs. long-distance trajectories
data_dist, data_dist_group = valspa.aggregate_data(data, ['subject', 'length'], aggregations.copy())
data_dist_sw_pre_post, data_dist_sw_pre_post_group = valspa.aggregate_data(data, ['subject', 'length', 'switch_pre_post'], aggregations.copy())

# active (choice) TP
data_act_tp, data_act_tp_group = valspa.aggregate_data(data, ['subject', 'active_timepoint'], aggregations.copy())

# angle (direction)
data_angle, data_angle_group = valspa.aggregate_data(data, ['subject', 'angle'], aggregations.copy())

# angle quadrant
data_angle_qu, data_angle_qu_group = valspa.aggregate_data(data, ['subject', 'angle_quadrant'], aggregations.copy())

# trajectories approximately parallel or perpendicular to the 45°-diagonal 
data_angles_diagonal_parallel_perpendicular_other, data_angles_diagonal_parallel_perpendicular_other_group = valspa.aggregate_data(data, ['subject', 'switch', 'angle_ref_diagonal'], aggregations.copy()) 


#%% EFFECT OF DISTANCE BETWEEN CHOICE LOCATION AND 45°-DIAGONAL ON PERFORMANCE: LOGISTIC REGRESSION

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import preprocessing

reg_to_scale = 'dist_to_diagonal'
mdl_formula = 'score ~ dist_to_diagonal_sc'
log_reg_per_sub = {} 

for sub in subs:
    log_reg_per_sub[sub] = {}
    
    # filter data according to trials
    log_reg_per_sub[sub]['data'] = copy.deepcopy(data_per_sub[sub])
    log_reg_per_sub[sub]['data'] = log_reg_per_sub[sub]['data'][~log_reg_per_sub[sub]['data']['score'].isnull()]
    # log_reg_per_sub[sub]['data'] = log_reg_per_sub[sub]['data'][log_reg_per_sub[sub]['data']['switch']==1]
    
    # scale regressor
    log_reg_per_sub[sub]['data'][f'{reg_to_scale}_sc'] = preprocessing.scale(log_reg_per_sub[sub]['data'][reg_to_scale])
    
    # logistic regression
    log_reg_per_sub[sub]['mdl'] = smf.glm(mdl_formula, data=log_reg_per_sub[sub]['data'], family=sm.families.Binomial()).fit()
    # print(log_reg_per_sub[sub]['mdl'].summary())
    log_reg_per_sub[sub]['mdl_params'] = log_reg_per_sub[sub]['mdl'].params[:].reset_index()
    log_reg_per_sub[sub]['mdl_params']['subject'] = sub[:]
    

# parameter coefficients across subjects
log_reg_coeff = pd.concat(log_reg_per_sub[sub]['mdl_params'] for sub in log_reg_per_sub)
log_reg_coeff = log_reg_coeff.rename(columns={'index': 'param', 0:'coeff'})
log_reg_coeff = log_reg_coeff.loc[log_reg_coeff['param']!='Intercept']

# check outlier: how far away from mean 
m = log_reg_coeff[log_reg_coeff.subject!='46'].coeff.mean()
s = log_reg_coeff[log_reg_coeff.subject!='46'].coeff.std()
x = log_reg_coeff.loc[log_reg_coeff.subject=='46', 'coeff'].item()
print((x-m)/s)
log_reg_coeff = log_reg_coeff[log_reg_coeff.subject!='46']

# save both for all trials and switch trials only
logistic_reg_dist_to_diagonal = copy.deepcopy(log_reg_coeff)
logistic_reg_dist_to_diagonal['param'] = 'all_trials'
# repeat logistic regression for switch trials only
logistic_reg_dist_to_diagonal_switch_trials = copy.deepcopy(log_reg_coeff)
logistic_reg_dist_to_diagonal_switch_trials['param'] = 'switch_trials_only'

logistic_reg_dist_to_diagonal_both = pd.concat([logistic_reg_dist_to_diagonal, logistic_reg_dist_to_diagonal_switch_trials])


# visualize effect by plotting correct and incorrect choice locations in space
data_incorrect_choice_locations = []
data_correct_choice_locations = []
data_all_choice_locations = []

for i_row in range(len(data)):
    act_tp = data.loc[i_row,'active_timepoint']
    score = data.loc[i_row,'score']   
    location = [data.loc[i_row,f'timepoint_{act_tp}_val_A'], data.loc[i_row,f'timepoint_{act_tp}_val_B']]
    data_all_choice_locations.append([location[0],location[1]])
    if score==0:
        data_incorrect_choice_locations.append([location[0],location[1]])
    elif score==1:
        data_correct_choice_locations.append([location[0],location[1]])


#%% EFFECT OF DISTANCE BETWEEN CHOICE LOCATION AND 45°-DIAGONAL ON REACTION TIME: LINEAR REGRESSION

reg_to_scale = 'dist_to_diagonal'
mdl_formula = 'rt_log ~ dist_to_diagonal_sc'
lin_reg_per_sub = {} 

for sub in subs:
    lin_reg_per_sub[sub] = {}
    
    # filter data according to trials
    lin_reg_per_sub[sub]['data'] = copy.deepcopy(data_per_sub[sub])
    lin_reg_per_sub[sub]['data'] = lin_reg_per_sub[sub]['data'][~lin_reg_per_sub[sub]['data']['score'].isnull()]
    # lin_reg_per_sub[sub]['data'] = lin_reg_per_sub[sub]['data'][lin_reg_per_sub[sub]['data']['switch']==1]
    
    # scale regressor
    lin_reg_per_sub[sub]['data'][f'{reg_to_scale}_sc'] = preprocessing.scale(lin_reg_per_sub[sub]['data'][reg_to_scale])
    
    # linear regression
    lin_reg_per_sub[sub]['mdl'] = smf.glm(mdl_formula, data=lin_reg_per_sub[sub]['data']).fit()
    # print(lin_reg_per_sub[sub]['mdl'].summary())
    lin_reg_per_sub[sub]['mdl_params'] = lin_reg_per_sub[sub]['mdl'].params[:].reset_index()
    lin_reg_per_sub[sub]['mdl_params']['subject'] = sub[:]
    

# parameter coefficients across subjects
lin_reg_coeff = pd.concat(lin_reg_per_sub[sub]['mdl_params'] for sub in lin_reg_per_sub)
lin_reg_coeff = lin_reg_coeff.rename(columns={'index': 'param', 0:'coeff'})
lin_reg_coeff = lin_reg_coeff.loc[lin_reg_coeff['param']!='Intercept']

# save both for all trials and switch trials only
rt_reg_dist_to_diagonal = copy.deepcopy(lin_reg_coeff)
rt_reg_dist_to_diagonal['param'] = 'all_trials'
# repeat linear regression for switch trials only
rt_reg_dist_to_diagonal_switch_trials = copy.deepcopy(lin_reg_coeff)
rt_reg_dist_to_diagonal_switch_trials['param'] = 'switch_trials_only'

rt_reg_dist_to_diagonal_both = pd.concat([rt_reg_dist_to_diagonal, rt_reg_dist_to_diagonal_switch_trials])


#%% REINFORCEMENT LEARNING MODEL

# Fit a reinforcement learning model which captures the prospective
# nature of the task to the choice data: modified Rescorla-Wagner model. 


"""
MODEL FUNCTIONS
"""

def rescorla_wagner_original(outcome, value, alpha):
    """
    Baseline: original Rescorla-Wagner model updates values over TPs in a trial
    according to a prediction error 

    Parameters
    ----------
    outcome : outcome received at current TP
    value : value estimate (before value update)
    alpha : learning rate

    Returns
    -------
    value : new value estimate after update   
    
    """
    
    pe = outcome - value # prediction error
    value = value + alpha * pe
    return value



def rescorla_wagner_prospective_target(outcome, value, alpha, outcomes, i_tp):   
    """
    Modified prospective Rescorla-Wagner model which updates according to prediction error 
    and values changes over TPs (target is the expected outcome)

    Parameters
    ----------
    outcome : outcome received at current TP
    value : value estimate (before value update)
    alpha : learning rate
    outcomes : outcomes of all TPs of the given trials
    i_tp : which TP

    Returns
    -------
    value : new value estimate after update  

    """
    
    change = outcome - outcomes[i_tp,:] # value change over TPs   
    value  = value + alpha * (outcome + change - value)      
    return value



# ALTERNATIVES FOR THE PROSPECTIVE MODEL:

def rescorla_wagner_prospective_target_expected_change(outcome, value, alpha, outcomes, i_tp, alpha_change, change_exp):
    """
    Prospective control model 1: 
    Similar to prospective Rescorla-Wagner model (rescorla_wagner_prospective_target), 
    but the option’s value change is updated itself over time points with its own learning rate

    Parameters
    ----------
    outcome : outcome received at current TP
    value : value estimate (before value update)
    alpha : learning rate
    outcomes : outcomes of all TPs of the given trials
    i_tp : which TP
    alpha_change : learning rate for value change
    change_exp : expected value change

    Returns
    -------
    value : new value estimate after update  
    change_exp : expected value change

    """
    
    change = outcome - outcomes[i_tp,:] # value change over TPs  
    
    # update change in later TPs
    if i_tp==0:
        change_exp = change*1
    else:
        change_exp = change_exp + alpha_change*change 
    
    value = value + alpha * (outcome + change_exp - value)  
        
    return value, change_exp



def rescorla_wagner_prospective_parameter(outcome, value, alpha, delta, outcomes, i_tp):
    """
    Prospective control model 2: 
    Value update with standard prediction error and an additional parameter for the value change

    Parameters
    ----------
    outcome : outcome received at current TP
    value : value estimate (before value update)
    alpha : learning rate
    delta : additional parameter for value change
    outcomes : outcomes of all TPs of the given trials
    i_tp : which TP

    Returns
    -------
    value : new value estimate after update 

    """
    
    pe = outcome - value # prediction error
    change = outcome - outcomes[i_tp,:] # value change over TPs  
    value = value + alpha*pe + delta*change 
    return value
    


def rescorla_wagner_expected_pe_without_alpha_last_pe(outcome, value, alpha, pe_exp, i_tp):
    """
    Prospective control model 3: 
    Value update with standard prediction error and expected prediction error, 
    similar to expected prediction error models in Wittmann et al. (2016)

    Parameters
    ----------
    outcome : outcome received at current TP
    value : value estimate (before value update)
    alpha : learning rate
    pe_exp : expected prediction error
    i_tp : which TP

    Returns
    -------
    value : new value estimate after update 
    pe_exp : expected prediction error

    """
    
    pe = outcome - value # prediction error
    
    # update expected prediction error in later TPs
    if i_tp==0:
        pe_exp = pe*1
    else:
        pe_exp = pe_exp + alpha*(pe-pe_exp)
    
    value = value + alpha*pe + pe_exp 
    
    return value, pe_exp



def rescorla_wagner_expected_pe_separate_alphas_value_pe_exp(outcome, value, alpha, pe_exp, alpha_pe_exp, i_tp):
    """
    Prospective control model 4: 
    Similar to prospective control model 3, but the expected prediction error is updated 
    with its own learning rate

    Parameters
    ----------
    outcome : outcome received at current TP
    value : value estimate (before value update)
    alpha : learning rate
    pe_exp : expected prediction error
    alpha_pe_exp : learning rate for expected prediction error 
    i_tp : which TP

    Returns
    -------
    value : new value estimate after update 
    pe_exp : expected prediction error

    """
    
    pe = outcome - value # prediction error
    
    # update expected prediction error in later TPs
    if i_tp==0:
        pe_exp = pe*1
    else:
        pe_exp = pe_exp + alpha_pe_exp*(pe-pe_exp)
    
    value = value + alpha*pe + pe_exp 
    
    return value, pe_exp



from scipy.special import softmax
def softmax_transform(value, beta):
    """
    Softmax function to translate values to choice probabilities 

    Parameters
    ----------
    value : value estimate
    beta : inverse temperature indicating the determinacy of choices

    Returns
    -------
    choice probabilities

    """
    
    return softmax(beta*value)



"""
MODEL INFORMATION
"""

# create dict with information regarding the model parameters 
rlmodel_info = {}

rlmodel_info['rescorla_wagner_original'] = {}
rlmodel_info['rescorla_wagner_original']['name_param'] = ['alpha', 'beta'] 
rlmodel_info['rescorla_wagner_original']['n_param'] = len(rlmodel_info['rescorla_wagner_original']['name_param'])
rlmodel_info['rescorla_wagner_original']['param_guess'] = [0.1, 2] # initial guess for parameters (required for scipy optimization, but the parameters will be estimated)
rlmodel_info['rescorla_wagner_original']['param_bounds'] = ((0, 1), (0, 100)) # bounds for parameters 

rlmodel_info['rescorla_wagner_prospective_target'] = {}
rlmodel_info['rescorla_wagner_prospective_target']['name_param'] = ['alpha', 'beta'] 
rlmodel_info['rescorla_wagner_prospective_target']['n_param'] = len(rlmodel_info['rescorla_wagner_prospective_target']['name_param'])
rlmodel_info['rescorla_wagner_prospective_target']['param_guess'] = [0.1, 2] # initial guess for parameters (required for scipy optimization, but the parameters will be estimated)
rlmodel_info['rescorla_wagner_prospective_target']['param_bounds'] = ((0, 3), (0, 100)) # bounds for parameters (for alpha: no upper bound of 1 for prospective models due to observed ceiling effect) 

rlmodel_info['rescorla_wagner_prospective_target_expected_change'] = {}
rlmodel_info['rescorla_wagner_prospective_target_expected_change']['name_param'] = ['alpha', 'beta', 'alpha_change'] 
rlmodel_info['rescorla_wagner_prospective_target_expected_change']['n_param'] = len(rlmodel_info['rescorla_wagner_prospective_target_expected_change']['name_param'])
rlmodel_info['rescorla_wagner_prospective_target_expected_change']['param_guess'] = [0.1, 2, 0.1] # initial guess for parameters (required for scipy optimization, but the parameters will be estimated)
rlmodel_info['rescorla_wagner_prospective_target_expected_change']['param_bounds'] = ((0, 3), (0, 100), (0, 3)) # bounds for parameters (for alpha: no upper bound of 1 for prospective models due to observed ceiling effect) 

rlmodel_info['rescorla_wagner_prospective_parameter'] = {}
rlmodel_info['rescorla_wagner_prospective_parameter']['name_param'] = ['alpha', 'beta', 'delta'] 
rlmodel_info['rescorla_wagner_prospective_parameter']['n_param'] = len(rlmodel_info['rescorla_wagner_prospective_parameter']['name_param'])
rlmodel_info['rescorla_wagner_prospective_parameter']['param_guess'] = [0.1, 2, 0.1] # initial guess for parameters (required for scipy optimization, but the parameters will be estimated)
rlmodel_info['rescorla_wagner_prospective_parameter']['param_bounds'] = ((0, 3), (0, 100), (0, 3)) # bounds for parameters (for alpha: no upper bound of 1 for prospective models due to observed ceiling effect) 

rlmodel_info['rescorla_wagner_expected_pe_without_alpha_last_pe'] = {}
rlmodel_info['rescorla_wagner_expected_pe_without_alpha_last_pe']['name_param'] = ['alpha', 'beta'] 
rlmodel_info['rescorla_wagner_expected_pe_without_alpha_last_pe']['n_param'] = len(rlmodel_info['rescorla_wagner_expected_pe_without_alpha_last_pe']['name_param'])
rlmodel_info['rescorla_wagner_expected_pe_without_alpha_last_pe']['param_guess'] = [0.1, 2] # initial guess for parameters (required for scipy optimization, but the parameters will be estimated)
rlmodel_info['rescorla_wagner_expected_pe_without_alpha_last_pe']['param_bounds'] = ((0, 2), (0, 100)) # bounds for parameters (for alpha: no upper bound of 1 for prospective models due to observed ceiling effect, changed to 2 as optimize didn't work otherwise but ensured that it wouldn't cause ceiling) 

rlmodel_info['rescorla_wagner_expected_pe_separate_alphas_value_pe_exp'] = {}
rlmodel_info['rescorla_wagner_expected_pe_separate_alphas_value_pe_exp']['name_param'] = ['alpha', 'beta', 'alpha_pe_exp'] 
rlmodel_info['rescorla_wagner_expected_pe_separate_alphas_value_pe_exp']['n_param'] = len(rlmodel_info['rescorla_wagner_expected_pe_separate_alphas_value_pe_exp']['name_param'])
rlmodel_info['rescorla_wagner_expected_pe_separate_alphas_value_pe_exp']['param_guess'] = [0.1, 2, 0.1] # initial guess for parameters (required for scipy optimization, but the parameters will be estimated)
rlmodel_info['rescorla_wagner_expected_pe_separate_alphas_value_pe_exp']['param_bounds'] = ((0, 2), (0, 100), (0, 2)) # bounds for parameters (for alpha: no upper bound of 1 for prospective models due to observed ceiling effect, changed to 2 as optimize didn't work otherwise but ensured that it wouldn't cause ceiling)


"""
COMPUTE MODEL FUNCTION
"""

def compute_model(x, *args):
    """
    Function to compute model:
    can be used to either simulate data with prespecified parameters or to search 
    for best-fitting parameters by passing it on to an optimization function
    
    Logic:
    Task consists of independent trials (trajectories), i.e., values are updated
    according to a model within a trial over TPs (not across trials). Each trial 
    consists of multiple TPs: first an observation phase and then a choice (= active TP).

    Parameters
    ----------
    x : model parameters
    *args : 
        data
        mode: either 'loglike' (for optimization) or 'simulateData' (to extract estimated values etc.)
        model

    Returns
    -------
    TYPE
        DESCRIPTION.
    log_like : log likelihood of the model
    aic : Akaike Information Criterion of the model
    val_act_tp : model-derived values at the active TPs of all trials
    prob_act_tp : model-derived choice probabilities at the active TPs of all trials

    """
    
    # extract the inputs (possibly passed by scipy.optimize.minimize)
    data_rlm, mode, model = args
    
    if model=='rescorla_wagner_original':
        alpha, beta = x # model parameters
    elif model=='rescorla_wagner_prospective_target':
        alpha, beta = x 
    elif model=='rescorla_wagner_prospective_target_expected_change':
        alpha, beta, alpha_change = x
    elif model=='rescorla_wagner_prospective_parameter':
        alpha, beta, delta = x 
    elif model=='rescorla_wagner_expected_pe_without_alpha_last_pe':
        alpha, beta = x
    elif model=='rescorla_wagner_expected_pe_separate_alphas_value_pe_exp':
        alpha, beta, alpha_pe_exp = x 
    
    # initialize arrays to store model-derived values and choice probabilities for active TPs across trials
    val_act_tp  = np.zeros([len(data_rlm), 2])
    prob_act_tp = np.zeros([len(data_rlm), 2])

    # loop thorugh trials
    for i_trial in range(len(data_rlm)):
        
        # get all objective values (received outcomes) until active TP
        active_tp = int(data_rlm.loc[i_trial,'active_timepoint'])
        outcomes = np.zeros([active_tp,2])
        for i_tp in range(active_tp):
            outcomes[i_tp,0] = data_rlm.loc[i_trial, f'timepoint_{i_tp+1}_val_A']
            outcomes[i_tp,1] = data_rlm.loc[i_trial, f'timepoint_{i_tp+1}_val_B']

        # initialize arrays to store model-derived values and choice probabilities within given trial
        val_in_trial  = np.zeros([active_tp-1,2])
        prob_in_trial = np.zeros([active_tp-1,2])
        
        # initialize values with values of first TP in given trial
        value = outcomes[0,:]
        # initialize expected change and prediction error (will be updated over TPs)
        change_exp = 0
        pe_exp = 0 

        # loop through TPs: values were initialized with the objective values of first TP,
        # outcomes of the second TP will be received and value predictions will be made 
        # for the following TPs
        for i_tp in range(active_tp-1):

            # compute probabilities for choice options given their current values by
            # using the softmax transformation (values divided by 100 to make betas in
            # softmax comparable to estimates for probabilistic values in other tasks)
            prob_softmax = softmax_transform(value/100, beta)

            # save current values and choice probabilities
            val_in_trial[i_tp,:] = value[:]
            prob_in_trial[i_tp,:] = prob_softmax[:]
            # if this is the active TP, save current values and choice probabilities
            # of the active TP of given trial
            # (active_tp-2) = values to be expected at active TP before outcome is received
            # (active_tp-1) would be updated values after outcome at active TP was received
            if i_tp==active_tp-2:
                val_act_tp[i_trial,:] = value[:]
                prob_act_tp[i_trial,:] = prob_softmax[:]

            # update value according to model
            if model=='rescorla_wagner_original':
                value = eval(model + '(outcomes[i_tp+1,:], value, alpha)')
            elif model=='rescorla_wagner_prospective_target': 
                value = eval(model + '(outcomes[i_tp+1,:], value, alpha, outcomes, i_tp)')
            elif model=='rescorla_wagner_prospective_target_expected_change':
                result = eval(model + '(outcomes[i_tp+1,:], value, alpha, outcomes, i_tp, alpha_change, change_exp)')
                value = result[0]
                change_exp = result[1]
            elif model=='rescorla_wagner_prospective_parameter':
                value = eval(model + '(outcomes[i_tp+1,:], value, alpha, delta, outcomes, i_tp)')
            elif model=='rescorla_wagner_expected_pe_without_alpha_last_pe':
                result = eval(model + '(outcomes[i_tp+1,:], value, alpha, pe_exp, i_tp)')
                value = result[0]
                pe_exp = result[1]
            elif model=='rescorla_wagner_expected_pe_separate_alphas_value_pe_exp':
                result = eval(model + '(outcomes[i_tp+1,:], value, alpha, pe_exp, alpha_pe_exp, i_tp)')
                value = result[0]
                pe_exp = result[1]


    # calculate log likelihood of model given the model-derived probabilities for
    # the choice options and the actually chosen option
    log_like = np.sum(np.log(prob_act_tp[data_rlm['chosen_option']==1, 0])) + np.sum(np.log(prob_act_tp[data_rlm['chosen_option']==2, 1]))
    # calculate Akaike Information Criterion
    aic = 2*rlmodel_info[model]['n_param'] - 2*log_like
    # negative log likelihood for optimization function to find best fitting parameters (minimize)
    log_like = -log_like 
    
    
    # function returns depending on mode (for optimization return only log likelihood)
    if mode=='loglike':
        return log_like
    elif mode=='simulateData':
        return log_like, aic, val_act_tp, prob_act_tp



"""
RUN MODELS
"""
        
rlm_per_sub = {} 

for sub in subs:
    
    rlm_per_sub[sub] = {}
    
    # filter data according to trials
    rlm_per_sub[sub]['data'] = copy.deepcopy(data_per_sub[sub]) # copy to allow more flexibility when filtering
    rlm_per_sub[sub]['data'] = rlm_per_sub[sub]['data'][~rlm_per_sub[sub]['data']['score'].isnull()].reset_index(drop=True)
    
    # # quick plot of choice data, with green and red dots depicting correct and incorrect 
    # # responses respectively 
    # trials              = np.arange(len(rlm_per_sub[sub]['data']))
    # correct_trials      = trials[rlm_per_sub[sub]['data']['chosen_option']==rlm_per_sub[sub]['data']['correct_option']]
    # correct_choices     = rlm_per_sub[sub]['data']['chosen_option'][rlm_per_sub[sub]['data']['chosen_option']==rlm_per_sub[sub]['data']['correct_option']] # which option was chosen
    # incorrect_trials    = trials[rlm_per_sub[sub]['data']['chosen_option']!=rlm_per_sub[sub]['data']['correct_option']]
    # incorrect_choices   = rlm_per_sub[sub]['data']['chosen_option'][rlm_per_sub[sub]['data']['chosen_option']!=rlm_per_sub[sub]['data']['correct_option']]

    # plt.figure()
    # plt.scatter(correct_trials,correct_choices-1, color='green')
    # plt.scatter(incorrect_trials,incorrect_choices-1, color='red')
    # plt.title(f'Chosen option for subject {sub}')
    # plt.xlabel('Trial number')
    # plt.ylabel('Chosen option')
    # plt.yticks([0,1])
    # plt.show()

    # fit models by
    # 1. minimizing the negative log likelihood of a given model to find best-fitting parameters
    # 2. fit model with estimated parameters to extract model-derived values etc. 
    for mdl in rlmodel_info:
        
        rlm_per_sub[sub][mdl] = {}
        
        # Step 1: minimize log likelihood using the scipy optimize function
        rlm_per_sub[sub][mdl]['optimize_result'] = scipy.optimize.minimize(compute_model, 
                                                                           rlmodel_info[mdl]['param_guess'], 
                                                                           args=(rlm_per_sub[sub]['data'], 'loglike', mdl), 
                                                                           method='L-BFGS-B', 
                                                                           bounds=rlmodel_info[mdl]['param_bounds']) 
        
        # save best-fitting parameters
        rlm_per_sub[sub][mdl]['fitted_parameters'] = rlm_per_sub[sub][mdl]['optimize_result'].x
        
        # Step 2: fit model with estimated parameters to extract model-derived values etc. 
        sim_data = compute_model(rlm_per_sub[sub][mdl]['fitted_parameters'], *(rlm_per_sub[sub]['data'], 'simulateData', mdl))
        rlm_per_sub[sub][mdl]['neg_log_like'] = sim_data[0]
        rlm_per_sub[sub][mdl]['aic'] = sim_data[1]
        rlm_per_sub[sub][mdl]['mdl_deriv_val'] = sim_data[2]
        rlm_per_sub[sub][mdl]['mdl_deriv_prob'] = sim_data[3]
        
        # include model-derived values in subject-specific df
        data_per_sub[sub].loc[~data_per_sub[sub]['score'].isnull(), f'{mdl}_val_A'] = rlm_per_sub[sub][mdl]['mdl_deriv_val'][:,0]
        data_per_sub[sub].loc[~data_per_sub[sub]['score'].isnull(), f'{mdl}_val_B'] = rlm_per_sub[sub][mdl]['mdl_deriv_val'][:,1]
        
        data_per_sub[sub][f'{mdl}_val_chosen'] = data_per_sub[sub].apply(lambda x: x[f'{mdl}_val_A'] if x['chosen_option']==1
                                                                         else x[f'{mdl}_val_B'] if x['chosen_option']==2
                                                                         else np.nan,
                                                                         axis=1) 
        
        data_per_sub[sub][f'{mdl}_val_unchosen'] = data_per_sub[sub].apply(lambda x: x[f'{mdl}_val_A'] if x['chosen_option']==2
                                                                           else x[f'{mdl}_val_B'] if x['chosen_option']==1
                                                                           else np.nan,
                                                                           axis=1) 
        
        data_per_sub[sub][f'{mdl}_val_difference'] = data_per_sub[sub][f'{mdl}_val_chosen'] - data_per_sub[sub][f'{mdl}_val_unchosen']
    

"""
SAVE MODEL AIC AND PARAMETERS
"""

# AIC
aic_rlm = pd.DataFrame(columns=['subject', 'model', 'AIC'])
for sub in subs:
    for mdl in rlmodel_info:
        aic_rlm = aic_rlm.append({'subject': sub, 'model': mdl, 'AIC': rlm_per_sub[sub][mdl]['aic']}, ignore_index=True)

data_rlm_aic_all_models = copy.deepcopy(aic_rlm)
data_rlm_aic_rw_vs_rwpt = data_rlm_aic_all_models[data_rlm_aic_all_models.model.isin(['rescorla_wagner_original', 'rescorla_wagner_prospective_target'])].reset_index(drop=True)

# parameters alpha and beta for winning model
data_rlm_rwpt_params = pd.DataFrame(columns=['subject', 'alpha', 'beta'])
for sub in subs:
    data_rlm_rwpt_params = data_rlm_rwpt_params.append({'subject': sub, 'alpha': rlm_per_sub[sub]['rescorla_wagner_prospective_target']['fitted_parameters'][0], 'beta': rlm_per_sub[sub]['rescorla_wagner_prospective_target']['fitted_parameters'][1]}, ignore_index=True)

    for mdl in rlmodel_info:
        aic_rlm = aic_rlm.append({'subject': sub, 'alpha': rlm_per_sub[sub]['rescorla_wagner_prospective_target']['fitted_parameters'][0], 'beta': rlm_per_sub[sub]['rescorla_wagner_prospective_target']['fitted_parameters'][1]}, ignore_index=True)

# save subject dfs with model-derived values (also used for fMRI analysis)
if not os.path.isfile(os.path.join(valspa.behav_results_path, task, 'behav_analyzed_data_per_sub.pkl')):   
    with open(os.path.join(valspa.behav_results_path, task, 'behav_analyzed_data_per_sub.pkl'), 'wb') as f:
        pickle.dump(data_per_sub, f)
    


#%% SAVE BEHAVIORAL RESULTS

behav_data = {'data_overall': data_overall,
              'data_sw_pre_post': data_sw_pre_post,
              'data_dist_sw_pre_post': data_dist_sw_pre_post,
              'data_angle': data_angle,
              'data_angle_group': data_angle_group,
              'data_angle_qu': data_angle_qu,
              'data_angles_diagonal_parallel_perpendicular_other': data_angles_diagonal_parallel_perpendicular_other,
              'data_logistic_reg_dist_to_diag': logistic_reg_dist_to_diagonal_both,
              'data_rt_reg_dist_to_diag': rt_reg_dist_to_diagonal_both,
              'data_all_choice_locations': data_all_choice_locations,
              'data_correct_choice_locations': data_correct_choice_locations,
              'data_incorrect_choice_locations': data_incorrect_choice_locations,
              'data_rlm_aic_rw_vs_rwpt': data_rlm_aic_rw_vs_rwpt,
              'data_rlm_aic_all_models': data_rlm_aic_all_models,
              'data_rlm_rwpt_params': data_rlm_rwpt_params}


if not os.path.isfile(os.path.join(valspa.behav_results_path, task, 'behavioral_data.pkl')):   
    with open(os.path.join(valspa.behav_results_path, task, 'behavioral_data.pkl'), 'wb') as f:
        pickle.dump(behav_data, f)


#%% PAPER FIGURES AND STATISTICS

# Load new environment with updated patchworklib and scipy packages

import seaborn as sn
import patchworklib as pw
from plotnine.options import set_option
set_option('base_family',  'Arial')

# import behavioral data
with open(os.path.join(valspa.behav_results_path, 'DEC', 'behavioral_data.pkl'), 'rb') as f:
    behav = pickle.load(f)
    
# import SBSOD questionnaire  data
with open(os.path.join(valspa.behav_results_path, 'SBSOD', 'data_sbsod.pkl'), 'rb') as f:
    sbsod = pickle.load(f)

# plotting variables
plot_path = os.path.join(valspa.behav_results_path, task, 'paper')
font_size_axis_text = 8
font_size_axis_title = 9
# for raincloud plots: how much to shift the violin, points and lines
shift = 0.1
# figure size:
fs_stand = [6.4, 4.8]
# 180mm/3figures = 60 per fig, 60mm=2.36inch
fs_rs = (2.36, 1.77)  
fs_lw = (1.5,1.77)


"""
COMBINE BEHAVIORAL AND SBSOD DATA
"""

behav['behav_post_tasks_correlations'] = behav['data_overall'][['subject','score_mean']] 
behav['behav_post_tasks_correlations'].rename(columns={'score_mean': 'overall_score'}, inplace=True)

x = behav['data_rlm_rwpt_params'][['subject','alpha']] 
x.rename(columns={'alpha': 'rwpt_alpha'}, inplace=True)
behav['behav_post_tasks_correlations'] = x.merge(behav['behav_post_tasks_correlations'], on=['subject'])

x = sbsod[['subject','sbsod_score']] 
behav['behav_post_tasks_correlations'] = x.merge(behav['behav_post_tasks_correlations'], on=['subject'])


"""
BEHAVIOR MAIN FIGURE (PAPER: FIGURE 2)
"""

# plot individual panel figures and use patchworklib to combine all panels into one figure 

# FIGURE 1: Overall performance
fig1_m1 = aes(x=stage('measure', after_scale='x+shift'))    # shift outward
fig1_m2 = aes(x=stage('measure', after_scale='x-2*shift'))  # shift inward

fig1 = (ggplot(behav['data_overall'])
         + aes(x='measure', y='score_mean')
         + geom_violin(fig1_m1, style='right', fill=valspa.color_petrol, color=None)
         + geom_jitter(fig1_m2, width=shift, height=0, stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
         + scale_y_continuous(limits = [0.6,1.01], labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + labs(x='Overall performance', y='Percentage correct choices')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig1
ggsave(plot = fig1, filename = 'fig1.pdf', path = plot_path)
g1 = pw.load_ggplot(fig1, figsize=fs_rs)
g1.set_index('a', size=font_size_axis_title)


# FIGURE 2: Performance in switch trajectories (pre, switch, post)

# one-sample t-test against chance performance
for c in behav['data_sw_pre_post'].switch_pre_post.unique():
    print(f'Statistics for {c}')
    print(valspa.one_sample_permutation_t_test(behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post==c].score_mean - 0.5))
    print(behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post==c].score_mean.mean())
    print(behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post==c].score_mean.std())

# repeated measures ANOVA to test for differences between time points
aovrm = AnovaRM(behav['data_sw_pre_post'], 'score_mean', 'subject', within=['switch_pre_post'])
res = aovrm.fit()
print(res)
# post-hoc tests
x = behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post=='pre'].score_mean
y = behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post=='switch'].score_mean
valspa.paired_sample_permutation_t_test(x, y)

fig2_m1 = aes(x=stage('switch_pre_post', after_scale='x+shift'))
fig2_m2 = aes(x=stage('switch_pre_post', after_scale='x-shift'))
colors_sw_pre_post = {'pre':'#9EAA78', 'switch':'#77805A', 'post':'#4F553C'}

fig2 = (ggplot(behav['data_sw_pre_post'])
         + aes(x='switch_pre_post', y='score_mean', fill='switch_pre_post')
         + geom_violin(fig2_m1, style='right', color=None)
         + geom_point(fig2_m2, stroke=0, size=1.5)
         + geom_line(aes(group='subject'), color='grey', size=0.1, position=position_nudge(-0.1))
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
         + scale_y_continuous(limits = [0,1.02], labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + scale_x_discrete(limits = ['pre', 'switch', 'post'])
         + scale_fill_manual(values = colors_sw_pre_post, guide=False)
         + labs(x='Time point', y='Percentage correct choices')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x='switch', y=1, label='***', size=font_size_axis_text)
         + annotate(geom='text', x='pre', y=1, label='***', size=font_size_axis_text)
         + annotate(geom='text', x='post', y=1, label='***', size=font_size_axis_text)
        )
fig2 
ggsave(plot = fig2, filename = 'fig2.pdf', path = plot_path)
g2 = pw.load_ggplot(fig2, figsize=fs_rs)
g2.set_index('b', size=font_size_axis_title)


# FIGURE 3: Effect of distance between choice location and 45°-diagonal on performance
for c in behav['data_logistic_reg_dist_to_diag'].param.unique():
    print(f'Statistics for {c}')
    print(valspa.one_sample_permutation_t_test(behav['data_logistic_reg_dist_to_diag'][behav['data_logistic_reg_dist_to_diag'].param==c].coeff))
    print(behav['data_logistic_reg_dist_to_diag'][behav['data_logistic_reg_dist_to_diag'].param==c].coeff.mean())
    print(behav['data_logistic_reg_dist_to_diag'][behav['data_logistic_reg_dist_to_diag'].param==c].coeff.std())

fig3_m1 = aes(x=stage('param', after_scale='x+shift'))
fig3_m2 = aes(x=stage('param', after_scale='x-2*shift'))
color_dist_diag = '#0073D1'

fig3 = (ggplot(behav['data_logistic_reg_dist_to_diag'])
         + aes(x='param', y='coeff')
         + geom_violin(fig3_m1, style='right', fill=color_dist_diag, color=None)
         + geom_jitter(fig3_m2, width=shift, height=0, stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=color_dist_diag, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['all_trials', 'switch_trials_only'], labels = ['all trajectories', 'switch trajectories only'])
         + geom_hline(yintercept = 0)
         + labs(x='Type of trajectory', y='Effect size (a.u.)')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x='all_trials', y=behav['data_logistic_reg_dist_to_diag'].coeff.max()+0.1, label='***', size=font_size_axis_text)
         + annotate(geom='text', x='switch_trials_only', y=behav['data_logistic_reg_dist_to_diag'].coeff.max()+0.1, label='***', size=font_size_axis_text)
        )
fig3 # call figure 
ggsave(plot = fig3, filename = 'fig3.pdf', path = plot_path)
g3 = pw.load_ggplot(fig3, figsize=fs_rs)
g3.set_index('c', size=font_size_axis_title)


# FIGURE 4: Visualization of diagonal effect
# (plot separately as it doesn't work properly with patchworklib)
# g4 = pw.Brick(figsize=(6.4,4.8))
g4 = plt.figure()
g4 = plt.gca() # get the axis handle
g4.set(title = 'Value space', 
        xlabel = 'Value latent option A', 
        ylabel = 'Value latent option B', 
        xticks = valspa.dec.axis_ticks, 
        yticks = valspa.dec.axis_ticks,
        xlim = valspa.dec.axis_limits, 
        ylim = valspa.dec.axis_limits)
g4.set_axisbelow(True)
g4.grid(True)
g4.set_aspect('equal', adjustable='box')
# sn.kdeplot(x=behav['data_incorrect_choice_locations'][:,0], y=behav['data_incorrect_choice_locations'][:,1], fill=True, ax=g4)
# add diagonal
g4.arrow(valspa.dec.diag_p1[0], 
          valspa.dec.diag_p1[1], 
          valspa.dec.diag_p2[0]-valspa.dec.diag_p1[0], 
          valspa.dec.diag_p2[1]-valspa.dec.diag_p1[1],  
          head_width=0,
          head_length=0,
          color = 'red')
g4.scatter(behav['data_all_choice_locations'][:,0], behav['data_all_choice_locations'][:,1], color='grey', alpha=0.3, s=15, linewidths=0)
g4.scatter(behav['data_incorrect_choice_locations'][:,0], behav['data_incorrect_choice_locations'][:,1], color='blue', alpha=0.8, s=15, linewidths=0)
fig_name = 'fig4.pdf' 
plt.savefig(os.path.join(plot_path, fig_name))
plt.show()


# FIGURE 5: Reinforcement learning model comparison
valspa.paired_sample_permutation_t_test(behav['data_rlm_aic_rw_vs_rwpt'][behav['data_rlm_aic_rw_vs_rwpt'].model=='rescorla_wagner_prospective_target'].AIC,
                                        behav['data_rlm_aic_rw_vs_rwpt'][behav['data_rlm_aic_rw_vs_rwpt'].model=='rescorla_wagner_original'].AIC)

# function to allow placement of objects left-right
def alt_sign(x):
    "Alternate +1/-1 if x is even/odd"
    return (-1) ** x

fig5_m1 = aes(x=stage('model', after_scale='x+shift*alt_sign(x)'))            
fig5_m2 = aes(x=stage('model', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_rw_rwpt = {'rescorla_wagner_original':'#9E7D0A', 'rescorla_wagner_prospective_target':'#256D69'}

fig5 = (ggplot(behav['data_rlm_aic_rw_vs_rwpt'])
         + aes(x='model', y='AIC', fill='model')
         + geom_violin(fig5_m1, style='left-right', color=None)
         + geom_point(fig5_m2, stroke=0, size=1.5)
         + geom_line(fig5_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig5_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['rescorla_wagner_original', 'rescorla_wagner_prospective_target'], labels = ['Original RW Model', 'Prospective RW Model'])
         + scale_fill_manual(values = colors_rw_rwpt, guide=False)
         + labs(x='Reinforcement learning model', y='Akaike information criterion')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1.5, y=behav['data_rlm_aic_rw_vs_rwpt'].AIC.max()+2, label='***', size=font_size_axis_text)
        )
fig5 # call figure 
ggsave(plot = fig5, filename = 'fig5.pdf', path = plot_path)
g5 = pw.load_ggplot(fig5, figsize=fs_rs)
g5.set_index('e', size=font_size_axis_title)


# FIGURE 6: Correlation with SBSOD
corr = valspa.correlation_permutation_test(behav['behav_post_tasks_correlations'].rwpt_alpha, behav['behav_post_tasks_correlations'].sbsod_score)
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(2)}'

fig6 = (ggplot(behav['behav_post_tasks_correlations'])
         + aes(x='rwpt_alpha', y='sbsod_score')
         + geom_point(stroke=0, size=1.5)
         + geom_smooth(method='lm', color=colors_rw_rwpt['rescorla_wagner_prospective_target'], fill=colors_rw_rwpt['rescorla_wagner_prospective_target'])
         # + scale_x_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + labs(x='Learning rate of prospective RW model', y='Navigational abilities (SBSOD score)')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1.4, y=2.5, label=corr_txt, size=font_size_axis_text)           
        )
fig6
ggsave(plot = fig6, filename = 'fig6.pdf', path = plot_path)
g6 = pw.load_ggplot(fig6, figsize=fs_rs)
g6.set_index('f', size=font_size_axis_title)


# # create object for empty space for patchworklib to work 
# g7 = pw.load_ggplot(fig6, figsize=fs_rs)
# g7.set_index('d', size=font_size_axis_title)


# arrange all panels in one figure using patchworklib
pw.param["margin"]=0.3
g_all = (g1/g7)|(g2/g5)|(g3/g6)
g_all.savefig(os.path.join(plot_path,'behav_Fig2_composed.pdf'))



"""
BEHAVIOR SUPPLEMENTARY FIGURE (PAPER: SUPPLEMENTARY FIGURE 2)
"""

# FIGURE 1
# repeated measures ANOVA to test for differences between time points
aovrm = AnovaRM(behav['data_sw_pre_post'], 'rt_log_mean', 'subject', within=['switch_pre_post'])
res = aovrm.fit()
print(res)
# post-hoc tests
x = behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post=='pre'].rt_log_mean
y = behav['data_sw_pre_post'][behav['data_sw_pre_post'].switch_pre_post=='switch'].rt_log_mean
valspa.paired_sample_permutation_t_test(x, y)

supp_fig1_m1 = aes(x=stage('switch_pre_post', after_scale='x+shift'))
supp_fig1_m2 = aes(x=stage('switch_pre_post', after_scale='x-shift'))
# colors_sw_pre_post = {'pre':'#D8DBC8', 'switch':'#9EAA78', 'post':'#6A7A48'}

supp_fig1 = (ggplot(behav['data_sw_pre_post'])
             + aes(x='switch_pre_post', y='rt_log_mean', fill='switch_pre_post')
             + geom_violin(supp_fig1_m1, style='right', color=None)
             + geom_point(supp_fig1_m2, stroke=0, size=1.5)
             + geom_line(aes(group='subject'), color='grey', size=0.1, position=position_nudge(-0.1))
             + geom_boxplot(outlier_shape='', width=shift, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
             + scale_x_discrete(limits = ['pre', 'switch', 'post'])
             + geom_hline(yintercept = 0)
             + scale_fill_manual(values = colors_sw_pre_post, guide=False)
             + labs(x='Time point', y='log(RT)')
             + theme_classic()
             + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
            )
supp_fig1 
ggsave(plot = supp_fig1, filename = 'supp_fig1.pdf', path = plot_path)
supp_g1 = pw.load_ggplot(supp_fig1, figsize=fs_rs)
supp_g1.set_index('a', size=font_size_axis_title)


# FIGURE 2
for c in behav['data_rt_reg_dist_to_diag'].param.unique():
    print(f'Statistics for {c}')
    print(valspa.one_sample_permutation_t_test(behav['data_rt_reg_dist_to_diag'][behav['data_rt_reg_dist_to_diag'].param==c].coeff))
    
supp_fig2_m1 = aes(x=stage('param', after_scale='x+shift'))
supp_fig2_m2 = aes(x=stage('param', after_scale='x-2*shift'))
color_dist_diag = '#0073D1'

supp_fig2 = (ggplot(behav['data_rt_reg_dist_to_diag'])
             + aes(x='param', y='coeff')
             + geom_violin(supp_fig2_m1, style='right', fill=color_dist_diag, color=None)
             + geom_jitter(supp_fig2_m2, width=shift, height=0, stroke=0, size=1.5)
             + geom_boxplot(outlier_shape='', width=shift, fill=color_dist_diag, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
             + scale_x_discrete(limits = ['all_trials', 'switch_trials_only'], labels = ['all trajectories', 'switch trajectories only'])
             + geom_hline(yintercept = 0)
             + labs(x='Type of trajectory', y='Effect of distance from diagonal (a.u.)')
             + theme_classic()
             + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
            )
supp_fig2 # call figure 
ggsave(plot = supp_fig2, filename = 'supp_fig2.pdf', path = plot_path)
supp_g2 = pw.load_ggplot(supp_fig2, figsize=fs_rs)
supp_g2.set_index('b', size=font_size_axis_title)


# FIGURE 3
# repeated measures ANOVA 
aovrm = AnovaRM(behav['data_dist_sw_pre_post'], 'score_mean', 'subject', within=['length', 'switch_pre_post'])
res = aovrm.fit()
print(res)
# direct comparison of switch TP
valspa.paired_sample_permutation_t_test(behav['data_dist_sw_pre_post'][(behav['data_dist_sw_pre_post'].switch_pre_post=='switch') & (behav['data_dist_sw_pre_post'].length==30)].score_mean,
                                        behav['data_dist_sw_pre_post'][(behav['data_dist_sw_pre_post'].switch_pre_post=='switch') & (behav['data_dist_sw_pre_post'].length==50)].score_mean)

supp_fig3_m1 = aes(x=stage('switch_pre_post', after_scale='x+shift'))
supp_fig3_m2 = aes(x=stage('switch_pre_post', after_scale='x-shift'))
# colors_sw_pre_post = {'pre':'#D8DBC8', 'switch':'#9EAA78', 'post':'#6A7A48'}

supp_fig3 = (ggplot(behav['data_dist_sw_pre_post'])
             + aes(x='switch_pre_post', y='score_mean', fill='switch_pre_post')
             + geom_violin(supp_fig3_m1, style='right', color=None)
             + geom_point(supp_fig3_m2, stroke=0, size=1.5)
             + geom_line(aes(group='subject'), color='grey', size=0.1, position=position_nudge(-0.1))
             + geom_boxplot(outlier_shape='', width=shift, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
             + scale_y_continuous(limits = [0,1.02], labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
             + scale_x_discrete(limits = ['pre', 'switch', 'post'])
             + scale_fill_manual(values = colors_sw_pre_post, guide=False)
             + labs(x='Time point', y='Percentage correct choices')
             + facet_wrap('~length')
             + theme_classic()
             + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
            )
supp_fig3 
ggsave(plot = supp_fig3, filename = 'supp_fig3.pdf', path = plot_path)
supp_g3 = pw.load_ggplot(supp_fig3, figsize=fs_rs)
supp_g3.set_index('c', size=font_size_axis_title)


# FIGURE 4
supp_fig4_x_breaks = copy.deepcopy(behav['data_angle_group'].angle)
supp_fig4_x_labels = [''] * len(supp_fig4_x_breaks)
supp_fig4_x_labels[::2] = copy.deepcopy(behav['data_angle_group'].angle[::2])

supp_fig4 = (ggplot(behav['data_angle_group'])
            + aes(x='angle',y='score_mean_mean')
            + geom_line(color=valspa.color_petrol)
            + labs(x='Direction (angle)', y='Percentage correct choices')
            + scale_x_continuous(breaks = supp_fig4_x_breaks, labels = supp_fig4_x_labels)
            + scale_y_continuous(limits = [0.7, 1.01], labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
            + geom_errorbar(aes(ymin = behav['data_angle_group'].score_mean_mean - behav['data_angle_group'].score_mean_sem, ymax = behav['data_angle_group'].score_mean_mean + behav['data_angle_group'].score_mean_sem), width = 0.2)
            + theme_classic()
            + theme(axis_text_x = element_text(angle=90), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
            )
supp_fig4 
ggsave(plot = supp_fig4, filename = 'supp_fig4.pdf', path = plot_path)
supp_g4 = pw.load_ggplot(supp_fig4, figsize=fs_rs)
supp_g4.set_index('d', size=font_size_axis_title)


# FIGURE 5
# repeated measures ANOVA 
aovrm = AnovaRM(behav['data_angle_qu'], 'score_mean', 'subject', within=['angle_quadrant'])
res = aovrm.fit()
print(res)

behav['data_angle_qu']['angle_quadrant'] = behav['data_angle_qu']['angle_quadrant'].astype(str)
supp_fig5_m1 = aes(x=stage('angle_quadrant', after_scale='x+shift'))
supp_fig5_m2 = aes(x=stage('angle_quadrant', after_scale='x-2*shift'))

supp_fig5 = (ggplot(behav['data_angle_qu'])
             + aes(x='angle_quadrant', y='score_mean')
             + geom_violin(supp_fig5_m1, style='right', fill=valspa.color_petrol, color=None)
             + geom_jitter(supp_fig5_m2, width=shift, height=0, stroke=0, size=1.5)
             + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
             + scale_x_discrete(limits = ['1', '2', '3', '4', '0'], labels = ['Q1', 'Q2', 'Q3', 'Q4', 'cardinal'])
             + scale_y_continuous(limits = [0.5, 1.01], labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
             + labs(x='Quadrant of direction (angle)', y='Percentage correct choices')
             + theme_classic()
             + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
            )
supp_fig5
ggsave(plot = supp_fig5, filename = 'supp_fig5.pdf', path = plot_path)
supp_g5 = pw.load_ggplot(supp_fig5, figsize=fs_rs)
supp_g5.set_index('e', size=font_size_axis_title)


# FIGURE 6
# remove subjects with NaNs
behav['data_angles_diagonal_parallel_perpendicular_other'] = behav['data_angles_diagonal_parallel_perpendicular_other'][behav['data_angles_diagonal_parallel_perpendicular_other'].subject!='31']
behav['data_angles_diagonal_parallel_perpendicular_other'] = behav['data_angles_diagonal_parallel_perpendicular_other'][behav['data_angles_diagonal_parallel_perpendicular_other'].subject!='53']
# repeated measures ANOVA
aovrm = AnovaRM(behav['data_angles_diagonal_parallel_perpendicular_other'], 'score_mean', 'subject', within=['switch', 'angle_ref_diagonal'])
res = aovrm.fit()
print(res)
# post hoc tests
valspa.paired_sample_permutation_t_test(behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==0) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='parallel')].score_mean, 
                                        behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==0) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='perpendicular')].score_mean)

valspa.paired_sample_permutation_t_test(behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==0) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='parallel')].score_mean, 
                                        behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==0) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='other')].score_mean)

valspa.paired_sample_permutation_t_test(behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==0) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='perpendicular')].score_mean, 
                                        behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==0) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='other')].score_mean)

valspa.paired_sample_permutation_t_test(behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==1) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='parallel')].score_mean, 
                                        behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==1) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='perpendicular')].score_mean)

valspa.paired_sample_permutation_t_test(behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==1) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='parallel')].score_mean, 
                                        behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==1) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='other')].score_mean)

valspa.paired_sample_permutation_t_test(behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==1) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='perpendicular')].score_mean, 
                                        behav['data_angles_diagonal_parallel_perpendicular_other'][(behav['data_angles_diagonal_parallel_perpendicular_other'].switch==1) & (behav['data_angles_diagonal_parallel_perpendicular_other'].angle_ref_diagonal=='other')].score_mean)

supp_fig6_m1 = aes(x=stage('angle_ref_diagonal', after_scale='x+shift'))            
supp_fig6_m2 = aes(x=stage('angle_ref_diagonal', after_scale='x-shift'))

# re-order categories
behav['data_angles_diagonal_parallel_perpendicular_other']['switch'] = behav['data_angles_diagonal_parallel_perpendicular_other']['switch'].replace(0, 'non_switch') 
behav['data_angles_diagonal_parallel_perpendicular_other']['switch'] = behav['data_angles_diagonal_parallel_perpendicular_other']['switch'].replace(1, 'switch') 
behav['data_angles_diagonal_parallel_perpendicular_other']['switch'] = behav['data_angles_diagonal_parallel_perpendicular_other']['switch'].astype('category')
behav['data_angles_diagonal_parallel_perpendicular_other']['switch'] = behav['data_angles_diagonal_parallel_perpendicular_other']['switch'].cat.reorder_categories(['switch', 'non_switch'])

supp_fig6 = (ggplot(behav['data_angles_diagonal_parallel_perpendicular_other'])
         + aes(x='angle_ref_diagonal', y='score_mean')
         + geom_violin(supp_fig6_m1, style='right', fill=valspa.color_petrol, color=None)
         + geom_point(supp_fig6_m2, stroke=0, size=1.5)
         + geom_line(aes(group='subject'), color='grey', size=0.1, position=position_nudge(-0.1))
         + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['parallel', 'perpendicular', 'other'], labels = ['para.', 'perp.', 'other'])
         + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l])
         + labs(x='Direction referenced to diagonal', y='Percentage correct choices')
         + facet_wrap('~switch')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title)) 
        )
supp_fig6
ggsave(plot = supp_fig6, filename = 'supp_fig6.pdf', path = plot_path)
supp_g6 = pw.load_ggplot(supp_fig6, figsize=fs_rs)
supp_g6.set_index('f', size=font_size_axis_title)


# FIGURE 7
behav['data_rlm_rwpt_params']['measure'] = 'parameter'
supp_fig7_m1 = aes(x=stage('measure', after_scale='x+shift'))
supp_fig7_m2 = aes(x=stage('measure', after_scale='x-2*shift'))  

supp_fig7 = (ggplot(behav['data_rlm_rwpt_params'])
          + aes(x='measure', y='alpha')
          + geom_violin(supp_fig7_m1, style='right', fill=colors_rw_rwpt['rescorla_wagner_prospective_target'], color=None)
          + geom_jitter(supp_fig7_m2, width=shift, height=0, fill=colors_rw_rwpt['rescorla_wagner_prospective_target'], stroke=0, size=1.5)
          + geom_boxplot(outlier_shape='', width=shift, fill=colors_rw_rwpt['rescorla_wagner_prospective_target'], color='black')
          + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
          + scale_y_continuous(limits = [0,1.7], expand=[0,0])
          + labs(x='Learning rate alpha', y='Estimated parameter')
          + theme_classic()
          + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
supp_fig7
ggsave(plot = supp_fig7, filename = 'supp_fig7.pdf', path = plot_path)
supp_g7 = pw.load_ggplot(supp_fig7, figsize=fs_rs)
supp_g7.set_index('g', size=font_size_axis_title)


# FIGURE 8
behav['data_rlm_rwpt_params_perf'] = copy.deepcopy(behav['data_rlm_rwpt_params'])
y_s = copy.deepcopy(behav['data_sw_pre_post'].loc[behav['data_sw_pre_post'].switch_pre_post=='switch', ['subject','score_mean']])
behav['data_rlm_rwpt_params_perf'] = y_s.merge(behav['data_rlm_rwpt_params_perf'], on=['subject'])
behav['data_rlm_rwpt_params_perf'].rename(columns={'score_mean': 'switch_performance'}, inplace=True)
y_p = copy.deepcopy(behav['data_sw_pre_post'].loc[behav['data_sw_pre_post'].switch_pre_post=='pre', ['subject','score_mean']])
behav['data_rlm_rwpt_params_perf'] = y_p.merge(behav['data_rlm_rwpt_params_perf'], on=['subject'])
behav['data_rlm_rwpt_params_perf'].rename(columns={'score_mean': 'pre_performance'}, inplace=True)

corr = valspa.correlation_permutation_test(behav['data_rlm_rwpt_params_perf'].alpha, behav['data_rlm_rwpt_params_perf'].switch_performance)
corr_txt = f'r = {corr.statistic.round(2)}\np < .001'

supp_fig8 = (ggplot(behav['data_rlm_rwpt_params_perf'])
             + aes(x='alpha', y='switch_performance')
             + geom_point(stroke=0, size=1.5)
             + geom_smooth(method='lm', color=colors_rw_rwpt['rescorla_wagner_prospective_target'], fill=colors_rw_rwpt['rescorla_wagner_prospective_target'])
             + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
             + labs(x='Learning rate alpha', y='Performance at switch')
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x=1.5, y=0.2, label=corr_txt, size=font_size_axis_text)           
            )
supp_fig8
ggsave(plot = supp_fig8, filename = 'supp_fig8.pdf', path = plot_path)
supp_g8 = pw.load_ggplot(supp_fig8, figsize=fs_rs)
supp_g8.set_index('h', size=font_size_axis_title)


# FIGURE 9
corr = valspa.correlation_permutation_test(behav['data_rlm_rwpt_params_perf'].alpha, behav['data_rlm_rwpt_params_perf'].pre_performance)
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(3)}'

supp_fig9 = (ggplot(behav['data_rlm_rwpt_params_perf'])
             + aes(x='alpha', y='pre_performance')
             + geom_point(stroke=0, size=1.5)
             + geom_smooth(method='lm', color=colors_rw_rwpt['rescorla_wagner_prospective_target'], fill=colors_rw_rwpt['rescorla_wagner_prospective_target'])
             + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
             + labs(x='Learning rate alpha', y='Performance at pre')
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x=1.5, y=0.95, label=corr_txt, size=font_size_axis_text)           
            )
supp_fig9
ggsave(plot = supp_fig9, filename = 'supp_fig9.pdf', path = plot_path)
supp_g9 = pw.load_ggplot(supp_fig9, figsize=fs_rs)
supp_g9.set_index('i', size=font_size_axis_title)


# FIGURE 10
for m in behav['data_rlm_aic_all_models'].model.unique():
    print(f'Test vs. {m}')
    x = behav['data_rlm_aic_all_models'][behav['data_rlm_aic_all_models'].model=='rescorla_wagner_prospective_target'].AIC
    y = behav['data_rlm_aic_all_models'][behav['data_rlm_aic_all_models'].model==m].AIC
    print(valspa.paired_sample_permutation_t_test(x, y))

supp_fig10_m1 = aes(x=stage('model', after_scale='x+shift'))
supp_fig10_m2 = aes(x=stage('model', after_scale='x-2*shift'))

# sort according to AIC
aggregations = {
        'AIC': ['mean','std','sem'],
        }
behav['data_rlm_aic_all_models_group'] = behav['data_rlm_aic_all_models'].groupby('model').agg(aggregations)
behav['data_rlm_aic_all_models_group'].columns = ["_".join(x) for x in behav['data_rlm_aic_all_models_group'].columns.ravel()]
behav['data_rlm_aic_all_models_group'] = behav['data_rlm_aic_all_models_group'].reset_index()

x_order = list(behav['data_rlm_aic_all_models_group'].sort_values(by=['AIC_mean'], ascending=False).model)
x_labels = ['ORW', 'PC4', 'PC3', 'PC2', 'PC1', 'PRW']
colors_models = {}
for m in behav['data_rlm_aic_all_models_group'].model:
    colors_models[m] = valspa.color_petrol  
colors_models.update({'rescorla_wagner_original': colors_rw_rwpt['rescorla_wagner_original'], 'rescorla_wagner_prospective_target': colors_rw_rwpt['rescorla_wagner_prospective_target']})

supp_fig10 = (ggplot(behav['data_rlm_aic_all_models'])
             + aes(x='model', y='AIC', fill='model')
             + geom_violin(supp_fig10_m1, style='right', color=None)
             + geom_jitter(supp_fig10_m2, width=shift, height=0, stroke=0, size=1.5)
             + geom_boxplot(outlier_shape='', width=shift, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
             + scale_x_discrete(limits = x_order, labels = x_labels)
             + labs(x='Reinforcement learning model', y='Akaike information criterion')
             + scale_fill_manual(values = colors_models, guide=False)
             + theme_classic()
             + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
            )
supp_fig10
ggsave(plot = supp_fig10, filename = 'supp_fig10.pdf', path = plot_path)
supp_g10 = pw.load_ggplot(supp_fig10, figsize=fs_rs)
supp_g10.set_index('j', size=font_size_axis_title)
# # create object for empty space for patchworklib to work 
# supp_g11 = pw.load_ggplot(supp_fig10, figsize=fs_rs)
# supp_g11.set_index('k', size=font_size_axis_title)
# supp_g12 = pw.load_ggplot(supp_fig10, figsize=fs_rs)
# supp_g12.set_index('l', size=font_size_axis_title)


# arrange all panels in one figure using patchworklib
pw.param["margin"]=0.3
g_supp_all = (supp_g1/supp_g4/supp_g7/supp_g10)|(supp_g2/supp_g5/supp_g8/supp_g11)|(supp_g3/supp_g6/supp_g9/supp_g12)
g_supp_all.savefig(os.path.join(plot_path,'behav_supp_Fig2_composed.pdf'))














