#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# VALUE fMRI ANALYSIS
#
# Test for modulation of fMRI signal by value difference between options during
# choices. Values are derived from the prospective Rescorla-Wagner model (see
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
model_name  = 'value_analysis'
# model_name  = 'value_analysis_control_reaction_time'
# model_name  = 'value_analysis_control_correct_trials'
model_path  = os.path.join(valspa.first_level_path, task, model_name)
if not os.path.exists(model_path): os.makedirs(model_path)

# regressors for GLM (events of interest to model)
# indicate start of strings for which trial_type in events file will be filtered
regressors = [  
    'traject_pass_tp_visual_presentation',                          # trajectory regressor including initial/passive time points
    'choice_visual_presentation',                                   # choice time point regressor
    'choice_chosen_value_rescorla_wagner_prospective_target',       # choice time point regressor parametrically modulated by chosen value
    'choice_unchosen_value_rescorla_wagner_prospective_target',     # choice time point regressor parametrically modulated by unchosen value
    # 'choice_reaction_time_log',                                     # control: choice time point regressor parametrically modulated by reaction time log
    # 'choice_correct',                                               # control: choice time point regressor for correct trials
    'feedback_general'                                              # feedback regressor
    ]

# add regressors of no interest
regressors.extend(fmri.regressors_no_interest)

# define contrasts of interest: dict with contrast names and formulas 
contrast_info = {}
contrast_info['contrast_definitions'] = {
    'traject_pass_tp_presentation': 'traject_pass_tp_visual_presentation',
    'choice_visual_presentation': 'choice_visual_presentation',
    'choice_chosen_value': 'choice_chosen_value_rescorla_wagner_prospective_target',
    'choice_unchosen_value': 'choice_unchosen_value_rescorla_wagner_prospective_target',
    'choice_value_difference': 'choice_chosen_value_rescorla_wagner_prospective_target - choice_unchosen_value_rescorla_wagner_prospective_target',
    'choice_value_sum': 'choice_chosen_value_rescorla_wagner_prospective_target + choice_unchosen_value_rescorla_wagner_prospective_target',
    'feedback_presentation': 'feedback_general'     
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
        # v_onsets = events_run.loc[events_run.trial_type=='choice_chosen_value_rescorla_wagner_prospective_target', 'onset']
        # for o in v_onsets:
        #     if not ((events_run.trial_type=='choice_correct') & (events_run.onset==o)).any():
        #         events_run = events_run.drop(events_run[(events_run.trial_type=='choice_chosen_value_rescorla_wagner_prospective_target') & (events_run.onset==o)].index)
        #         events_run = events_run.drop(events_run[(events_run.trial_type=='choice_unchosen_value_rescorla_wagner_prospective_target') & (events_run.onset==o)].index)
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
    
# run randomise
fmri.run_second_level_randomise(valspa = valspa, 
                                subjects = valspa.subs, 
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

fig_file = os.path.join(sl_model_path, f'contrast-{fig_contrast}', 'figure-wholebrain-map-value-difference-corr.pdf')
nilearn.plotting.plot_stat_map(tmap_img_masked, 
                               output_file = fig_file, 
                               bg_img = os.path.join(valspa.derivatives_path, 'masks', 'mni', 'MNI152_T1_1mm_brain.nii.gz'), 
                               black_bg = False, 
                               cut_coords = (0,-24,40), 
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


#%% TIMECOURSE OF CHOSEN AND UNCHOSEN VALUE EFFECTS IN SIGNIFICANT VMPFC CLUSTER

from nilearn.maskers import NiftiMasker
from sklearn.linear_model import LinearRegression

roi      = 'value_difference_vmpfc_cluster'
roi_file = os.path.join(valspa.derivatives_path, 'masks', 'mni', f'{roi}.nii.gz')

# specify interval relative to choice (as event of interest) for signal extraction
tc_start, tc_end, tc_steps = -1, 15, 0.1
tc_times = np.arange(tc_start, tc_end, tc_steps)

# load behavioral data for trial-related information
with open(os.path.join(valspa.behav_results_path, task, 'behav_analyzed_data_per_sub.pkl'), 'rb') as f:
    behav_sub = pickle.load(f)
   
# initialize lists for columns in df across subjects
subject     = []
time        = []
condition   = []
beta        = []


for sub in valspa.subs:
    
    # create directory for sub
    if not os.path.exists(os.path.join(valspa.derivatives_path, 'value_effect', f'sub-{sub}')): os.makedirs(os.path.join(valspa.derivatives_path, 'value_effect', f'sub-{sub}'))
    
    # initialize arrays to store betas for value effects
    betas_chosen_sub = np.empty((task_info.n_runs,len(tc_times)))
    betas_unchosen_sub = np.empty((task_info.n_runs,len(tc_times)))   
    
    
    for i_run, run in enumerate(task_info.runs):
        
        # fMRI run file
        run_file = layout.get(subject=sub, task=task, run=run, space=space_analysis, scope='fMRIPrep', suffix='bold', extension='nii.gz', return_type='file')[0]                   
        n_tr     = nib.load(run_file).shape[-1] # number of trs 
        
        
        """
        Extract timecourse of entire run
        """
        
        tc_file = os.path.join(valspa.derivatives_path, 'value_effect', f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_roi-{roi}_data-timecourse.pkl')
        
        if os.path.isfile(tc_file):
            with open(tc_file, 'rb') as f:
                timecourse_trs = pickle.load(f)

        else:               
            # initialize NiftiMasker for signal extraction
            masker = NiftiMasker(mask_img = roi_file,
                                  smoothing_fwhm = valspa.fmri.smoothing,
                                  standardize = 'zscore',
                                  standardize_confounds = False,
                                  high_pass = None, 
                                  t_r = valspa.fmri.tr)
            
            # confounds for run
            confounds_run = pd.read_csv(layout.get(subject=sub, task=task, run=run, scope='fMRIPrep', extension='.tsv')[0].path, sep='\t')
            confounds_run = valspa.fmri.preprocess_confounds(confounds_run, valspa.fmri.confound_regressors)
                              
            # extract timecourse, average over voxels of roi and save
            timecourse = masker.fit_transform(run_file, confounds=confounds_run)
            timecourse = timecourse.mean(axis=1)
            timecourse_trs  = pd.DataFrame(data=timecourse, index = np.arange(n_tr)*valspa.fmri.tr + valspa.fmri.tr*valspa.fmri.slice_time_reference)
            
            with open(tc_file, 'wb') as f:
                pickle.dump(timecourse_trs, f)
        
        
        """
        Run regression for each choice (trial) interval
        """
        
        # interpolate time course
        timecourse_intp = scipy.interpolate.interp1d(timecourse_trs.index, timecourse_trs[0], kind = 'cubic') # same as InterpolatedUnivariateSpline
        
        # get choice information 
        choice_info = behav_sub[sub][behav_sub[sub].run==run][['rescorla_wagner_prospective_target_val_chosen', 'rescorla_wagner_prospective_target_val_unchosen', 'trial_run', 'choice_screen_start']]
        choice_info = choice_info.dropna()
        
        # for each choice extract signal within interval
        choice_tc_times = np.array([choice + tc_times for choice in choice_info.choice_screen_start])
        tc_signal = timecourse_intp(choice_tc_times)

        # design matrix including chosen & unchosen value and trial number
        dm = choice_info.drop('choice_screen_start', axis=1)
        dm = dm - dm.mean()
        
        # regression
        betas_chosen_run, betas_unchosen_run  = [], []
        
        for i_time in range(len(tc_times)):
            y = tc_signal[:,i_time].reshape(-1,1)
            reg = LinearRegression().fit(dm, y) 
            betas_chosen_sub[i_run, i_time] = reg.coef_[0][0]
            betas_unchosen_sub[i_run, i_time] = reg.coef_[0][1]
    
    
    # add info to lists for df across subjects
    subject.extend([sub] * len(tc_times))
    time.extend(tc_times)
    condition.extend(['chosen'] * len(tc_times))
    beta.extend(np.mean(betas_chosen_sub,axis=0))
    subject.extend([sub] * len(tc_times))
    time.extend(tc_times)
    condition.extend(['unchosen'] * len(tc_times))
    beta.extend(np.mean(betas_unchosen_sub,axis=0))
    
    # # plot single subject
    # plt.plot(tc_times, np.mean(betas_chosen_sub,axis=0), color='b', label = 'chosen value')
    # plt.plot(tc_times, np.mean(betas_unchosen_sub,axis=0), color='r', label = 'unchosen value')
    # plt.title(f'Value effect timecourse for choices in {roi} in subject {sub}')
    # plt.xlabel('Time in sec locked to choice onset')
    # plt.ylabel('beta')
    # plt.legend()
    # # plt.savefig(os.path.join(valspa.derivatives_path, 'value_effect', f'sub-{sub}', f'sub-{sub}_task-{task}_roi-{roi}_figure-value-effect-timecourse.pdf'))
    # plt.show()


# df for effect across subjects
betas_df = pd.DataFrame({'subject': subject,
                         'time': time,
                         'condition': condition,
                         'beta': beta})

# save file
value_data = {'data_timecourse_vmpfc': betas_df}

with open(os.path.join(valspa.derivatives_path, 'value_effect', 'group', 'value_data.pkl'), 'wb') as f:
    pickle.dump(value_data, f)     



    
    
  
            
            
           
        
        
        


        
       
        
       
                          
        
        
