#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# CHOICE DECODING fMRI ANALYSIS
# STEP 2: GLM TO ESTIMATE NEURAL ACTIVATION PATTERNS OF CHOICE TIME POINTS IN 
# PROSPECTIVE DECISION MAKING TASK
#
# Estimated neural activation patterns of choice time points will be the test
# data for the subsequent choice decoding analysis.
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
model_name  = 'choice_time_points'
model_path  = os.path.join(valspa.first_level_path, task, model_name)
if not os.path.exists(model_path): os.makedirs(model_path)

# regressors for GLM (events of interest to model)
# indicate start of strings for which trial_type in events file will be filtered
regressors = [  
    'traject_pass_tp_visual_presentation',                          # trajectory regressor including initial/passive time points
    'choice_trial',                                                 # choice time point regressor for each trial separately
    'feedback_general'                                              # feedback regressor
    ]

# add regressors of no interest
regressors.extend(fmri.regressors_no_interest)

# space for first-level analysis (MNI or subject native space)
space_analysis = copy.deepcopy(fmri.space_subject)

# # wholebrain masks of subjects for analysis
# masks_subs = [] 


#%% RUN FIRST-LEVEL GLM

for sub in valspa.subs:
    
    print(f'Run first-level GLM for subject {sub}')
    
    # create first-level directory for sub
    if not os.path.exists(os.path.join(model_path,f'sub-{sub}')): os.makedirs(os.path.join(model_path,f'sub-{sub}'))
    
    # anatomical whole-brain mask for subject with functional resolution
    mask_anat = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_func = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space=space_analysis, suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_wholebrain = nilearn.image.resample_to_img(mask_anat, mask_func, interpolation='nearest')
    # masks_subs.append(mask_wholebrain)
   
    for run in task_info.runs:
        
        # events for run
        events_run = pd.read_csv(layout.get(subject=sub, task=task, run=run, suffix='events')[0].path, sep='\t')
        # filter by events / regressors of interest for GLM 
        events_run = events_run[events_run['trial_type'].str.startswith(tuple(regressors))] 
        
        # define contrasts of interest: dict with contrast names and formulas 
        choice_trials = [s for s in events_run.trial_type.unique() if s.startswith('choice')]
        contrast_info = {}
        contrast_info['contrast_definitions'] = {choice:choice for choice in choice_trials} 
        # prefix for contrast filenames 
        contrast_info['contrast_prefix'] = os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}')
        
        # run glm for run
        fmri.run_first_level_glm(bids_layout = layout, 
                                 sub = sub,
                                 task = task, 
                                 runs = run, 
                                 space_analysis = space_analysis, 
                                 events = events_run, 
                                 mask = mask_wholebrain, 
                                 smoothing = 0, # 0 for MVPA
                                 contrast_info = contrast_info)
        
      
        
