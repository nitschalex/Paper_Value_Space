#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# CHOICE DECODING fMRI ANALYSIS
# STEP 3: PERFORM DECODING
# 
# Test whether the high-value option is represented more strongly than the 
# low-value option during choices. 
# For this purpose, train a decoder (support vector classifier) on occipital-
# temporal cortex voxels to distinguish neural activation patterns of the four 
# category-specific stimuli (faces, tools, scenes, body parts). Then apply this 
# decoder to neural activation patterns of choice time points in the prospective 
# decision making task.
#
# Decoding analysis is based on scikit-learn, see resources:  
# - https://scikit-learn.org/stable/
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

import nilearn
import nilearn.masking
import nilearn.maskers
import re
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score


#%% SET VARIABLES

# training and test data/tasks for decoding
task_train = 'PVT'
task_test  = 'DEC'

# paths
model_path_train = os.path.join(valspa.first_level_path, task_train, 'stimulus_trials')
model_path_test  = os.path.join(valspa.first_level_path, task_test, 'choice_time_points')
mask_path        = os.path.join(valspa.derivatives_path, 'masks')
decoding_path    = os.path.join(valspa.derivatives_path, 'choice_decoding')
if not os.path.exists(decoding_path): os.makedirs(decoding_path)

# space for analysis (MNI or subject native space)
space_analysis = copy.deepcopy(valspa.fmri.space_subject)

# roi for analysis
roi_feature_selection = 'occipital_temporal_lobe_thr25' # roi for further feature selection based on PVT training data
roi_decoding_final    = f'{roi_feature_selection}_fs_final'

# number of permutations for permutation tests
n_permutations = 1000

# import behavioral data
with open(os.path.join(valspa.behav_results_path, 'DEC', 'behav_analyzed_data_per_sub.pkl'), 'rb') as f:
    behav = pickle.load(f)


#%% CREATE FINAL SUBJECT-SPECIFIC DECODING ROI MASKS BASED ON PVT TRAINING DATA

# intersect occipital-temporal cortex mask with gray matter mask and select 
# those voxels (features) which show the strongest univariate response to stimuli
# in the training data (PVT)
# (= feature selection for category-stimuli-responsive voxels/features)

# percentile of highest scores for feature selection  
fs_percentile = 20

n_voxels_final_masks = []

for sub in valspa.subs:
    
    # create subject folder
    if not os.path.exists(os.path.join(decoding_path, f'sub-{sub}')): os.makedirs(os.path.join(decoding_path, f'sub-{sub}'))
        
    # masks 
    mask_anat = os.path.join(mask_path, f'sub-{sub}', 'func', f'sub-{sub}_space-T1w_desc-wholebrain_mask.nii.gz')
    mask_gm   = os.path.join(mask_path, f'sub-{sub}', 'func', f'sub-{sub}_space-T1w_desc-gray_matter_mask.nii.gz')
    mask_roi  = os.path.join(mask_path, f'sub-{sub}', 'func', f'sub-{sub}_space-T1w_desc-{roi_feature_selection}_mask.nii.gz')
    mask_sub  = nilearn.masking.intersect_masks([mask_anat, mask_gm, mask_roi], threshold=1, connected=False)    
    n_voxels_final_masks.append((mask_sub.get_fdata()==1).sum())

    
    """
    LOAD TRIAL-WISE DATA (CONDITION LABELS AND FEATURE (VOXEL) VALUES)
    """
    
    # list of condition nii maps: trial-wise stimulus effects
    condition_maps_list = glob.glob(os.path.join(model_path_train, f'sub-{sub}', f'sub-{sub}_task-{task_train}_*_contrast-trial*_z_score.nii.gz'))
    # remove first trial to ensure equal number of trials per condition 
    condition_maps_list = [c for c in condition_maps_list if not 'trial_1_' in c] 
    # list of condition labels 
    condition_labels = [re.search('image_(.+?)_z_score', c).group(1) for c in condition_maps_list] 

    # mask condition nii maps and transform the data into n_samples x n_features (trials x voxels)
    nii_maps_masker     = nilearn.maskers.NiftiMasker(mask_img=mask_sub, standardize=True)
    train_decoding_data = nii_maps_masker.fit_transform(condition_maps_list)
    
    
    """
    PERFORM UNIVARIATE FEATURE SELECTION AND CREATE FINAL MASK FOR DECODING
    """
    
    # get indices of best features 
    feature_selection           = SelectPercentile(f_classif, percentile=fs_percentile).fit(train_decoding_data, condition_labels)
    feature_selection_best_idx  = feature_selection.get_feature_names_out(np.arange(np.shape(train_decoding_data)[1]))
    feature_selection_best_idx  = feature_selection_best_idx.astype(int)
    
    # reshape indices to create nii mask
    feature_selection_mask_data = np.zeros(np.shape(train_decoding_data)[1])
    feature_selection_mask_data[feature_selection_best_idx] = 1
    
    # save final mask 
    feature_selection_mask = nilearn.masking.unmask(feature_selection_mask_data, mask_sub)
    feature_selection_mask.to_filename(os.path.join(decoding_path, f'sub-{sub}', f'sub-{sub}_roi-{roi_decoding_final}_mask.nii.gz'))


# # average number of voxels of final decoding masks 
# print(np.mean(n_voxels_final_masks))
# print(np.std(n_voxels_final_masks))


#%% CONTROL ANALYSIS: DECODE STIMULUS CATEGORY IN PVT TRAINING DATA 

# Check decoding accuracy for stimulus category in PVT training data, using
# a cross-validation scheme with left-out trials as test set. 

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
      
    # mask 
    mask_sub = os.path.join(decoding_path, f'sub-{sub}', f'sub-{sub}_roi-{roi_decoding_final}_mask.nii.gz')

    
    """
    LOAD TRIAL-WISE DATA (CONDITION LABELS AND FEATURE (VOXEL) VALUES)
    """
    
    # list of condition nii maps: trial-wise stimulus effects
    condition_maps_list = glob.glob(os.path.join(model_path_train, f'sub-{sub}', f'sub-{sub}_task-{task_train}_*_contrast-trial*_z_score.nii.gz'))
    # remove first trial to ensure equal number of trials per condition 
    condition_maps_list = [c for c in condition_maps_list if not 'trial_1_' in c] 
    # list of condition labels 
    condition_labels = [re.search('image_(.+?)_z_score', c).group(1) for c in condition_maps_list] 

    # create "artificial" sessions for leave-trials out cross-validation: 
    # first, let each session contain one trial per condition (4 trials, 14 sessions)
    # (count how often stimulus was already in in condition_labels until given element)
    session_labels = [condition_labels[:i].count(condition_labels[i]) + 1 for i in range(len(condition_labels))] 
    # now combine two sessions into one to have more trials per session (8 trials, 7 sessions)
    session_labels = [int(np.ceil(i/2)) for i in session_labels]
    
    # mask condition nii maps and transform the data into n_samples x n_features (trials x voxels)
    nii_maps_masker     = nilearn.maskers.NiftiMasker(mask_img=mask_sub, standardize=False) # standardization later during cross-validation
    train_decoding_data = nii_maps_masker.fit_transform(condition_maps_list)   

    
    """
    PERFORM DECODING USING CROSS-VALIDATION (LEAVE-TRIALS-OUT)
    """
    
    # create decoding pipeline with scaler (standardization) and support vector classifier
    scaler_svc = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

    # initialize cross-validation object: leave one group=session of trials out 
    cv = LeaveOneGroupOut() 
    
    # perform decoding with cross-validation
    cv_scores = cross_val_score(scaler_svc, 
                                train_decoding_data, 
                                condition_labels, 
                                cv=cv, 
                                groups=session_labels)
    
    # combine cross-validation with permutation testing (permute condition labels
    # to compare original score to null distribution) 
    permutation_result = permutation_test_score(scaler_svc, 
                                                train_decoding_data, 
                                                condition_labels, 
                                                cv=cv, 
                                                groups=session_labels, 
                                                n_permutations=n_permutations)
    
    permutation_zscore = (permutation_result[0] - np.append(permutation_result[1], permutation_result[0]).mean()) / np.append(permutation_result[1], permutation_result[0]).std()
    
    # check scores separately for each condition
    condition_scorers = {}
    for cond in valspa.pvt.images:
        condition_scorers[cond] = make_scorer(recall_score, average = None, labels = [cond])
    cv_scores_per_cond = cross_validate(scaler_svc, 
                                        train_decoding_data, 
                                        condition_labels, 
                                        scoring=condition_scorers, 
                                        cv=cv, 
                                        groups=session_labels, 
                                        return_train_score=False)
    
    
    """
    SAVE RESULTS FOR SUBJECT
    """
    
    sub_results = {'accuracy': cv_scores.mean(),
                   'z_score': permutation_zscore}
    for cond in valspa.pvt.images:
        sub_results[f'accuracy_{cond}'] = cv_scores_per_cond[f'test_{cond}'].mean()
    
    for r in sub_results:       
        subject.append(sub)
        condition.append(r)
        effect_size.append(sub_results[r])
           

data_pvt_category_decoding = pd.DataFrame({'subject': subject,
                                           'condition': condition,
                                           'effect_size': effect_size})


# print(data_pvt_category_decoding[data_pvt_category_decoding.condition=='accuracy'].effect_size.mean())
# print(data_pvt_category_decoding[data_pvt_category_decoding.condition=='accuracy'].effect_size.std())


#%% CHOICE DECODING DURING PROSPECTIVE DECISION MAKING TASK

# Test whether the high-value option is represented more strongly than the 
# low-value option during choices. 
# Train the decoder using the PVT training data. Apply the decoder to choices
# of the decision making task. For each trial, the decoder will output probabilities
# associated with the four stimuli. Compare these probabilities, i.e., probability
# of the high-value vs. the low-value stimulus. Compare separately for stimuli
# on screen during choices, and for congruent off-screen stimuli.

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
    
    # mask 
    mask_sub = os.path.join(decoding_path, f'sub-{sub}', f'sub-{sub}_roi-{roi_decoding_final}_mask.nii.gz')

    
    """
    TRAIN DECODER USING PVT TRAINING DATA
    """
    
    # list of condition nii maps: trial-wise stimulus effects
    train_condition_maps_list = glob.glob(os.path.join(model_path_train, f'sub-{sub}', f'sub-{sub}_task-{task_train}_*_contrast-trial*_z_score.nii.gz'))
    # remove first trial to ensure equal number of trials per condition 
    train_condition_maps_list = [c for c in train_condition_maps_list if not 'trial_1_' in c] 
    # list of condition labels 
    train_condition_labels = [re.search('image_(.+?)_z_score', c).group(1) for c in train_condition_maps_list] 
    
    # mask condition nii maps and transform the data into n_samples x n_features (trials x voxels)
    nii_maps_masker     = nilearn.maskers.NiftiMasker(mask_img=mask_sub, standardize=True)
    train_decoding_data = nii_maps_masker.fit_transform(train_condition_maps_list)
    
    # train decoder
    choice_decoder = SVC(probability=True, random_state=42)
    choice_decoder.fit(train_decoding_data, train_condition_labels)
    
    
    """
    APPLY DECODER TO CHOICES IN PROSPECTIVE DECISION MAKING TASK
    """
    
    # load subject's behavioral data
    sub_behav = behav[sub].dropna(subset='score').copy()
    
    # create dict for stimulus congruencies
    stimulus_congruencies = {}
    stimulus_congruencies[sub_behav.loc[0,'timepoint_1_img_A'][:-6]] = copy.deepcopy(sub_behav.loc[0,'timepoint_2_img_A'][:-6])
    stimulus_congruencies[sub_behav.loc[0,'timepoint_2_img_A'][:-6]] = copy.deepcopy(sub_behav.loc[0,'timepoint_1_img_A'][:-6])
    stimulus_congruencies[sub_behav.loc[0,'timepoint_1_img_B'][:-6]] = copy.deepcopy(sub_behav.loc[0,'timepoint_2_img_B'][:-6])
    stimulus_congruencies[sub_behav.loc[0,'timepoint_2_img_B'][:-6]] = copy.deepcopy(sub_behav.loc[0,'timepoint_1_img_B'][:-6])
    
    # define high and low value stimuli in choice trials
    sub_behav['decod_high_value_stim'] = sub_behav.apply(lambda x: x[f'timepoint_{x.active_timepoint}_img_A'][:-6] if x.correct_option==1
                                                         else x[f'timepoint_{x.active_timepoint}_img_B'][:-6] if x.correct_option==2
                                                         else np.nan,
                                                         axis=1) 
    
    sub_behav['decod_low_value_stim']  = sub_behav.apply(lambda x: x[f'timepoint_{x.active_timepoint}_img_A'][:-6] if x.correct_option==2
                                                         else x[f'timepoint_{x.active_timepoint}_img_B'][:-6] if x.correct_option==1
                                                         else np.nan,
                                                         axis=1) 
    
    sub_behav['decod_congruent_high_value_stim'] = [stimulus_congruencies[s] for s in sub_behav['decod_high_value_stim']] 
    sub_behav['decod_congruent_low_value_stim']  = [stimulus_congruencies[s] for s in sub_behav['decod_low_value_stim']] 
    
    # define choice nii file
    sub_behav['decod_choice_nii_file'] = sub_behav.apply(lambda x: os.path.join(model_path_test, f'sub-{sub}', f'sub-{sub}_task-{task_test}_run-{x.run}_space-{space_analysis}_contrast-choice_trial_{x.trial_total}_z_score.nii.gz'),
                                                         axis=1)
    
    # mask choice nii maps, standardize across trials within run and transform 
    # the data into n_samples x n_features (trials x voxels)
    test_decoding_data = [nii_maps_masker.fit_transform(sub_behav[sub_behav.run==run].decod_choice_nii_file) for run in sub_behav.run.unique()]
    test_decoding_data = np.vstack(test_decoding_data)
    
    # apply decoder and extract probabilties assigned to the four stimuli
    decoder_probabilities = choice_decoder.predict_proba(test_decoding_data)
    
    # assign these original stimulus probabilities to the conditions of interest
    # (e.g. what is the probability of the high-value stimulus)
    for ci in sub_behav.columns[sub_behav.columns.str.endswith('stim')]:
        idx = [np.where(choice_decoder.classes_==s)[0][0] for s in sub_behav[ci]]
        sub_behav[f'{ci}_prob'] = decoder_probabilities[np.arange(decoder_probabilities.shape[0]),idx]
    
    
    """
    PERMUTATION TEST TO COMPARE AGAINST CHANCE LEVEL PERFORMANCE OF DECODER 
    (PERMUTE TRAINING LABELS AND RECOMPUTE PROBABILITIES)
    """
    
    # save original difference scores of high- vs. low-value stimuli and 
    # initialize arrays for permutation scores
    result_perm_high_low = np.empty((decoder_probabilities.shape[0], n_permutations))
    result_perm_high_low[:,0] = sub_behav.decod_high_value_stim_prob - sub_behav.decod_low_value_stim_prob 
    
    result_perm_congruent_high_low = np.empty((decoder_probabilities.shape[0], n_permutations))
    result_perm_congruent_high_low[:,0] = sub_behav.decod_congruent_high_value_stim_prob - sub_behav.decod_congruent_low_value_stim_prob 
    
    # make permutations reproducible by initializing random number generator to create seed states 
    perm_glob_rng = np.random.default_rng(seed=42)
    # use seed sequence to create n_perm independent random states
    perm_random_states = perm_glob_rng.bit_generator._seed_seq.spawn(n_permutations)
    
    # copy df to use for permutation testing
    sub_behav_perm = sub_behav[sub_behav.columns[sub_behav.columns.str.contains('stim')]].copy()
    
    for perm in range(n_permutations-1):
        
        # permute training labels        
        train_condition_labels_perm = np.random.default_rng(seed=perm_random_states[perm]).permutation(train_condition_labels)
        
        # train decoder with permuted labels
        choice_decoder.fit(train_decoding_data, train_condition_labels_perm)
        
        # apply decoder and extract probabilties assigned to the four stimuli
        decoder_probabilities = choice_decoder.predict_proba(test_decoding_data)
        
        # assign these stimulus probabilities to the conditions of interest
        # (e.g. what is the probability of the high-value stimulus)
        for ci in sub_behav_perm.columns[sub_behav_perm.columns.str.endswith('stim')]:
            idx = [np.where(choice_decoder.classes_==s)[0][0] for s in sub_behav_perm[ci]]
            sub_behav_perm[f'{ci}_prob'] = decoder_probabilities[np.arange(decoder_probabilities.shape[0]),idx]
        
        # compute difference scores of high- vs. low-value stimuli based on permuted data
        result_perm_high_low[:,perm+1] = sub_behav_perm.decod_high_value_stim_prob - sub_behav_perm.decod_low_value_stim_prob 
        result_perm_congruent_high_low[:,perm+1] = sub_behav_perm.decod_congruent_high_value_stim_prob - sub_behav_perm.decod_congruent_low_value_stim_prob 
    
    
    # convert permutation results to trial-wise z-scores
    sub_behav['decod_high_vs_low_value_stim_prob_zscore'] = np.apply_along_axis(lambda x: (x[0] - x.mean())/x.std(), 1, result_perm_high_low)
    sub_behav['decod_congruent_high_vs_low_value_stim_prob_zscore'] = np.apply_along_axis(lambda x: (x[0] - x.mean())/x.std(), 1, result_perm_congruent_high_low)
    
    
    """
    SAVE RESULTS PER SUBJECT
    """
    
    # # control analysis: effect only in switch trials
    # sub_behav = sub_behav[sub_behav.switch_pre_post=='switch']
    # # control analysis: effect only in incorrect trials
    # sub_behav = sub_behav[sub_behav.score==0]
    
    sub_results = {'probability_high_value_stimulus': sub_behav.decod_high_value_stim_prob.mean(),
                   'probability_low_value_stimulus': sub_behav.decod_low_value_stim_prob.mean(),
                   'probability_congruent_high_value_stimulus': sub_behav.decod_congruent_high_value_stim_prob.mean(),
                   'probability_congruent_low_value_stimulus': sub_behav.decod_congruent_low_value_stim_prob.mean(),
                   'probability_high_vs_low_value_stimulus_zscore': sub_behav.decod_high_vs_low_value_stim_prob_zscore.mean(),
                   'probability_congruent_high_vs_low_value_stimulus_zscore': sub_behav.decod_congruent_high_vs_low_value_stim_prob_zscore.mean()}
    
    for r in sub_results:       
        subject.append(sub)
        condition.append(r)
        effect_size.append(sub_results[r])


# df across subjects
data_choice_decoding_all_trials = pd.DataFrame({'subject': subject,
                                                'condition': condition,
                                                'effect_size': effect_size}) 

# data_choice_decoding_switch_trials = pd.DataFrame({'subject': subject,
#                                                    'condition': condition,
#                                                    'effect_size': effect_size}) 

# data_choice_decoding_incorrect_trials = pd.DataFrame({'subject': subject,
#                                                       'condition': condition,
#                                                       'effect_size': effect_size})  


#%% SAVE DECODING RESULTS

decoding_data = {'data_pvt_category_decoding': data_pvt_category_decoding,
                 'data_choice_decoding_all_trials': data_choice_decoding_all_trials,
                 'data_choice_decoding_switch_trials': data_choice_decoding_switch_trials,
                 'data_choice_decoding_incorrect_trials': data_choice_decoding_incorrect_trials}

with open(os.path.join(decoding_path, 'group', 'decoding_data.pkl'), 'wb') as f:
    pickle.dump(decoding_data, f) 






    
    
