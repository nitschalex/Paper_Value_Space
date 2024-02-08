#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# CHOICE DECODING fMRI ANALYSIS
# STEP 1: GLM TO ESTIMATE NEURAL ACTIVATION PATTERNS OF STIMULI (CATEGORY-
# SPECIFIC PICTURES) IN PICTURE VIEWING TASK
#
# Estimated neural activation patterns of stimuli will be the training data
# for the subsequent choice decoding analysis: train decoder to distinguish 
# between the four category-specific stimuli.
#
# Implement Least-Squares Separate (LSS) GLM approach (single-trial GLMs) for
# PVT to reduce collinearity between successive trials.  
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
task        = 'PVT'
task_info   = eval(f'valspa.{task.lower()}()')

# GLM model name
model_name  = 'stimulus_trials'
model_path  = os.path.join(valspa.first_level_path, task, model_name)
if not os.path.exists(model_path): os.makedirs(model_path)

# regressors for GLM (events of interest to model)
# indicate start of strings for which trial_type in events file will be filtered
regressors = [  
    'trial',       # trial regressor (all individual trials)
    'face_test',   # test trial regressor (later define a general test trial regressor for all stimuli)
    'body_test',   # test trial regressor
    'scene_test',  # test trial regressor
    'tool_test'    # test trial regressor
    ]

# add regressors of no interest
regressors.extend(fmri.regressors_no_interest)

# space for first-level analysis (MNI or subject native space)
space_analysis = copy.deepcopy(fmri.space_subject)

# # wholebrain masks of subjects for analysis
# masks_subs = [] 


#%% FUNCTION FOR LEAST SQUARES SEPARATE GLM

def lss_per_sub(sub):
    """
    Implement Least-Squares Separate (LSS) GLM approach for a given subject. 
    This means that a separate GLM will be computed for each trial of interest.
    One regressor models the given trial of interest and one regressor models
    all other trials.
    This function will be passed on to joblib.Parallel to compute GLMs for 
    subjects in parallel. 
    
    Parameters
    ----------
    sub : subject ID

    """
    
    # create first-level directory for sub
    if not os.path.exists(os.path.join(model_path,f'sub-{sub}')): os.makedirs(os.path.join(model_path,f'sub-{sub}'))
    
    # anatomical whole-brain mask for subject with functional resolution
    mask_anat = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_func = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space=space_analysis, suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_wholebrain = nilearn.image.resample_to_img(mask_anat, mask_func, interpolation='nearest')
   
    for run in task_info.runs:
            
        # events for run
        events_run = pd.read_csv(layout.get(subject=sub, task=task, run=run, suffix='events')[0].path, sep='\t')
        # filter by events / regressors of interest for GLM 
        events_run = events_run[events_run['trial_type'].str.startswith(tuple(regressors))] 
        # redefine stimulus-specific test trials as one general test trial regressor
        events_run.loc[events_run['trial_type'].str.contains('test'), 'trial_type'] = 'test_trials'      
    
        # list of stimulus trials of interest for single-trial glms
        trial_list = events_run.loc[events_run['trial_type'].str.contains('image'), 'trial_type']
        
        
        for trial in trial_list:
            
            # redefine events so that there is one regressor for the given trial
            # of interest and one regressor for all other trials
            events_single_trial = copy.deepcopy(events_run)                
            events_single_trial.loc[(events_single_trial['trial_type'].str.contains('image')) & (events_single_trial['trial_type']!=trial), 'trial_type'] = 'other_trials'
            
            # define contrasts of interest: dict with contrast names and formulas 
            contrast_info = {}
            contrast_info['contrast_definitions'] = {trial:trial}
            # prefix for contrast filenames 
            contrast_info['contrast_prefix'] = os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}')
            
            # run glm 
            fmri.run_first_level_glm(bids_layout = layout, 
                                     sub = sub,
                                     task = task, 
                                     runs = run, 
                                     space_analysis = space_analysis, 
                                     events = events_single_trial, 
                                     mask = mask_wholebrain, 
                                     smoothing = 0, # 0 for MVPA
                                     contrast_info = contrast_info)


#%% RUN GLMS IN PARALLEL

from joblib import Parallel, delayed

Parallel(n_jobs=10, prefer='threads')(delayed(lss_per_sub)(
           sub) for sub in valspa.subs
        )
                  
        
        
