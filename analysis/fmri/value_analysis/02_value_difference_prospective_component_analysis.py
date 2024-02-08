#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# PROSPECTIVE VALUE COMPONENT fMRI ANALYSIS
#
# Test for modulation of fMRI signal by prospective component of the value 
# difference between options during choices. Prospective component refers to 
# influence of values derived from the prospective Rescorla-Wagner model over
# values derived from non-prospective (original) Rescorla-Wagner model (see
# model fit in behavior).
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
import valspa.fmri_functions as fmri

import nibabel as nib
import nilearn
import nilearn.plotting


#%% SET VARIABLES FOR FIRST-LEVEL GLM

# bids data layout
layout      = BIDSLayout(valspa.bids_path, derivatives=True)

# task-relevant variables 
task        = 'DEC'
task_info   = eval(f'valspa.{task.lower()}()')

# GLM model name
model_name  = 'value_prospective_component_analysis'
# model_name  = 'value_prospective_component_analysis_control_reaction_time'
# model_name  = 'value_prospective_component_analysis_control_correct_trials'
# model_name  = 'value_prospective_component_analysis_control_distance_to_diagonal'
model_path  = os.path.join(valspa.first_level_path, task, model_name)
if not os.path.exists(model_path): os.makedirs(model_path)

# regressors for GLM (events of interest to model)
# indicate start of strings for which trial_type in events file will be filtered
regressors = [  
    'traject_pass_tp_visual_presentation',                          # trajectory regressor including initial/passive time points
    'choice_visual_presentation',                                   # choice time point regressor
    'choice_chosen_value_rwpt_rw_diff',                             # choice time point regressor parametrically modulated by prospective component of chosen value
    'choice_unchosen_value_rwpt_rw_diff',                           # choice time point regressor parametrically modulated by prospective component of unchosen value
    # 'choice_reaction_time_log',                                     # control: choice time point regressor parametrically modulated by reaction time log
    # 'choice_correct',                                               # control: choice time point regressor for correct trials
    # 'choice_distance_to_diagonal',                                  # control: choice time point regressor parametrically modulated by distance between choice location and 45°-diagonal
    'feedback_general'                                              # feedback regressor
    ]

# add regressors of no interest
regressors.extend(fmri.regressors_no_interest)

# define contrasts of interest: dict with contrast names and formulas 
contrast_info = {}
contrast_info['contrast_definitions'] = {
    'choice_chosen_value_rwpt_rw_diff': 'choice_chosen_value_rwpt_rw_diff',
    'choice_unchosen_value_rwpt_rw_diff': 'choice_unchosen_value_rwpt_rw_diff',
    'choice_value_difference': 'choice_chosen_value_rwpt_rw_diff - choice_unchosen_value_rwpt_rw_diff'
    }

# space for first-level analysis (MNI or subject native space)
space_analysis = copy.deepcopy(fmri.space_standard)

# wholebrain masks of subjects for analysis
masks_subs = [] 


#%% RUN FIRST-LEVEL GLM

for sub in valspa.subs:
    
    print(f'Run first-level GLM for subject {sub}')
    
    # create first-level directory for sub
    if not os.path.exists(os.path.join(model_path,f'sub-{sub}')): os.makedirs(os.path.join(model_path,f'sub-{sub}'))
    
    # anatomical whole-brain mask for subject with functional resolution
    mask_anat = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', space=space_analysis, suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_func = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space=space_analysis, suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_wholebrain = nilearn.image.resample_to_img(mask_anat, mask_func, interpolation='nearest')
    masks_subs.append(mask_wholebrain)
    
    # get events of interest for GLM to create regressors 
    events = []
    for run in task_info.runs:
        
        # events for run
        events_run = pd.read_csv(layout.get(subject=sub, task=task, run=run, suffix='events')[0].path, sep='\t')
        # filter by events / regressors of interest for GLM 
        events_run = events_run[events_run['trial_type'].str.startswith(tuple(regressors))]   
        
        # # control analysis: restrict value regressors to correct trials
        # v_onsets = events_run.loc[events_run.trial_type=='choice_chosen_value_rwpt_rw_diff', 'onset']
        # for o in v_onsets:
        #     if not ((events_run.trial_type=='choice_correct') & (events_run.onset==o)).any():
        #         events_run = events_run.drop(events_run[(events_run.trial_type=='choice_chosen_value_rwpt_rw_diff') & (events_run.onset==o)].index)
        #         events_run = events_run.drop(events_run[(events_run.trial_type=='choice_unchosen_value_rwpt_rw_diff') & (events_run.onset==o)].index)
        #     if ((events_run.trial_type=='choice_correct') & (events_run.onset==o)).any():
        #         events_run = events_run.drop(events_run[(events_run.trial_type=='choice_visual_presentation') & (events_run.onset==o)].index) # correct trials have their own regressor
        # events_run.trial_type = events_run.trial_type.replace('choice_visual_presentation', 'choice_rest') # separate regressor for remaining incorrect trials
                
        events.append(events_run)
    
    
    # prefix for contrast filenames 
    contrast_info['contrast_prefix'] = os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_analysis}')
    
    # run glm for all runs at once
    fmri.run_first_level_glm(bids_layout = layout, 
                             sub = sub,
                             task = task, 
                             runs = task_info.runs, 
                             space_analysis = space_analysis, 
                             events = events, 
                             mask = mask_wholebrain, 
                             smoothing = fmri.smoothing, 
                             contrast_info = contrast_info)
        

#%% RUN SECOND-LEVEL MODEL USING FSL RANDOMISE

# second-level contrasts of interest
sl_contrasts = list(contrast_info['contrast_definitions'].keys())
sl_contrasts = ['contrast-' + c for c in sl_contrasts]

# masks for second-level analysis
sl_masks = {'wholebrain': f'wholebrain_space-{fmri.space_standard}'}

# masks for second-level analysis: create intersection of all subjects masks
# so that analysis is performend only in voxels shared across subjects 
if not os.path.exists(os.path.join(valspa.second_level_path, task, 'masks')): os.makedirs(os.path.join(valspa.second_level_path, task, 'masks'))
sl_wholebrain_mask = os.path.join(valspa.second_level_path, task, 'masks', f'wholebrain_space-{fmri.space_standard}.nii.gz')
if not os.path.isfile(sl_wholebrain_mask):
    mask_img = nilearn.masking.intersect_masks(masks_subs, threshold=1, connected=False)
    mask_img.to_filename(sl_wholebrain_mask)
    
# subjects (all except for outlier)
sl_subjects = copy.deepcopy(valspa.subs)
sl_subjects.remove('06')
    
# run randomise
fmri.run_second_level_randomise(valspa = valspa, 
                                subjects = sl_subjects, 
                                task = task, 
                                model_name = model_name, 
                                second_level_masks = sl_masks,
                                second_level_contrasts = sl_contrasts)


#%% PLOT SECOND-LEVEL RESULTS WHOLEBRAIN MAP FOR VALUE DIFFERENCE VALUE 
# t-map thresholded at pFWE < .05

fig_contrast = 'choice_value_difference'
sl_model_path = os.path.join(valspa.second_level_path, task, model_name)

# create mask of TFCE images for pFWE < .05 and mask the t-map accordingly 
tmap_file       = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{fmri.space_standard}_tstat1.nii.gz')
pos_tfce_file   = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{fmri.space_standard}_tfce_corrp_tstat1.nii.gz')
neg_tfce_file   = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'negative_wholebrain_space-{fmri.space_standard}_tfce_corrp_tstat1.nii.gz')
pos_tfce_thr    = nilearn.image.math_img('img>=0.975', img=pos_tfce_file)
neg_tfce_thr    = nilearn.image.math_img('img>=0.975', img=neg_tfce_file)
comb_tfce_mask  = nilearn.image.math_img('img1 + img2', img1=pos_tfce_thr, img2=neg_tfce_thr)

tmap_img_masked = nilearn.image.math_img('img1 * img2', img1=tmap_file, img2=comb_tfce_mask)
tmap_img_masked.to_filename(os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{fmri.space_standard}_tstat1_masked_both_positive_negative.nii.gz')) 

fig_file = os.path.join(sl_model_path, f'contrast-{fig_contrast}', 'figure-wholebrain-map-value-difference-prospective-component-corr.pdf')
nilearn.plotting.plot_stat_map(tmap_img_masked, 
                               output_file = fig_file, 
                               bg_img = os.path.join(valspa.derivatives_path, 'masks', 'mni', 'MNI152_T1_1mm_brain.nii.gz'), 
                               black_bg = False, 
                               cut_coords = (-6,49,3), 
                               draw_cross = False)


#%% CLUSTER INFORMATION SECOND-LEVEL RESULTS WHOLEBRAIN MAP 

# get cluster information using nilearn 
clusters_table = nilearn.reporting.get_clusters_table(stat_img = tmap_img_masked, 
                                                      stat_threshold = 0.001,
                                                      two_sided = True,
                                                      min_distance = 0)

# add atlas labels using FSL atlasquery
fmri.get_second_level_cluster_labels(clusters_table = clusters_table, 
                                     output_path = os.path.join(sl_model_path, f'contrast-{fig_contrast}'))


    
                          
        
        
