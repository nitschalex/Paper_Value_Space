#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# HEXADIRECTIONAL fMRI ANALYSIS
# (ANALYSIS OF GRID-LIKE REPRESENTATION)
#
# Test for hexadirectional modulation of fMRI signal as a proxy measure for 
# grid cell activity (Doeller et al., 2010). This script implements the 
# hexadirectional analysis based on trajectories through an abstract value space. 
#
# Logic of the analysis:  
# 1. Estimate the putative grid orientation (i.e., phase of the hexadirectional
#    signal) using one part of the data  
# 2. Test for hexadirectional modulation aligned to the putative orientation
#    in left-out test data  
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
model_name  = 'hexadirectional_analysis'
model_path  = os.path.join(valspa.first_level_path, task, model_name)
if not os.path.exists(model_path): os.makedirs(model_path)

# regressors for GLM (events of interest to model)
# indicate start of strings for which trial_type in events file will be filtered
# analysis is based on cross-validation with different regressors for estimation
# (GLM1) and test (GLM2) data 

# regressors for estimation data (GLM1)
regressors_estimation = [   
    'traject_choice_visual_presentation',   # trajectory regressor including all time points
    'traject_choice_sin_n_fold',            # trajectory regressor parametrically modulated by sine of direction (angle) θ of trajectory with 60° (6-fold symmetry or changed accordingly for control symmetries) 
    'traject_choice_cos_n_fold',            # trajectory regressor parametrically modulated by cosine of direction (angle) θ of trajectory with 60° (6-fold symmetry or changed accordingly for control symmetries) 
    'feedback_general'                      # feedback regressor
    ]

# regressors for test set (GLM2)
# for test data, we run two GLMs:
# 1. main: GLM with a parametric regressor to test for hexadirectional modulation 
# 2. for visualization: GLM with separate regressors for bins of aligned and misaligned directions
regressors_test_glm_parametric = [   
    'traject_choice_visual_presentation',   # trajectory regressor including all time points
    'parametric_regressor',                 # trajectory regressor reflecting a 6-fold (hexadirectional, or control symmetry) sinusoidal modulation aligned to putative grid orientation
    'feedback_general'                      # feedback regressor  
    ]

regressors_test_glm_dir_bins = [
    'traject_choice_visual_presentation',   # trajectory regressor including all time points
    'traject_direction_bin',                # multiple regressors for direction bins (will be defined later)
    'feedback_general'                      # feedback regressor   
    ]

# add regressors of no interest
regressors_estimation.extend(fmri.regressors_no_interest)
regressors_test_glm_parametric.extend(fmri.regressors_no_interest)
regressors_test_glm_dir_bins.extend(fmri.regressors_no_interest)

# symmetries (n-fold modulation of fMRI signal): 
# hexadirectional=6-fold symmetry of interest, additionally control symmetries 
symmetries = np.arange(4,9) 
# symmetries = [6] 

# hypothesized directional peaks of 6-fold effect relative to arbitrary
# grid orientation of 0° 
hypo_dir_peaks_aligned = task_info.angles[::6]
hypo_dir_peaks_misaligned = hypo_dir_peaks_aligned + 30

# ROI used to estimate grid orientation 
roi = 'EC_Freesurfer' # subject-specific entorhinal cortex masks

# define contrasts of interest for estimation and test data (GLMs): dict with contrast names and formulas 
contrast_info = {}
for sym in symmetries:
    contrast_info[f'sym-{sym}'] = {}
    
    contrast_info[f'sym-{sym}']['estimation_glm'] = {}
    contrast_info[f'sym-{sym}']['estimation_glm']['contrast_definitions'] = {
        f'traject_choice_sin_{sym}_fold': f'traject_choice_sin_{sym}_fold',
        f'traject_choice_cos_{sym}_fold': f'traject_choice_cos_{sym}_fold'            
        }
    
    contrast_info[f'sym-{sym}']['test_glm_parametric'] = {}
    contrast_info[f'sym-{sym}']['test_glm_parametric']['contrast_definitions'] = {
        f'parametric_regressor_{sym}_fold': f'parametric_regressor_{sym}_fold'         
        }
    
    contrast_info[f'sym-{sym}']['test_glm_dir_bins'] = {}
    contrast_info[f'sym-{sym}']['test_glm_dir_bins']['contrast_definitions'] = {}
    for peak_bin in hypo_dir_peaks_aligned:                    
        contrast_info[f'sym-{sym}']['test_glm_dir_bins']['contrast_definitions'].update({f'direction_bin_{peak_bin}': f'traject_direction_bin_{peak_bin}'})
    for peak_bin in hypo_dir_peaks_misaligned:                    
        contrast_info[f'sym-{sym}']['test_glm_dir_bins']['contrast_definitions'].update({f'direction_bin_{peak_bin}': f'traject_direction_bin_{peak_bin}'}) 
    

# space for analysis (MNI or subject native space)
space_analysis  = copy.deepcopy(fmri.space_subject)
space_transform = copy.deepcopy(fmri.space_standard)

# wholebrain masks of subjects for analysis
masks_subs = [] 


#%% FUNCTIONS FOR ESTIMATION OF GRID ORIENTATION

def pol2cart(theta, rho):
    """Convert polar coordinates to cartesian coordinates"""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def cart2pol(x, y):
    """Convert cartesian coordinates to polar coordinates"""
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def estimate_orientation(sin_beta, cos_beta, sym):
    """
    Estimate grid orientation (i.e., phase of the signal) of each voxel and
    average across voxels with weighting by voxels' amplitudes

    Parameters
    ----------
    sin_beta : voxel-wise values of sine regressor of estimation GLM
    cos_beta : voxel-wise values of cosine regressor of estimation GLM
    sym : n-fold symmetry / modulation 

    Returns
    -------
    orient_mean : mean orientation in n-fold space (of ROI, radians) 
    orient_voxels_sym : voxel-wise orientations in n-fold space (radians) 
    amplitude_voxels : voxel-wise amplitudes of n-fold modulation

    """

    # calculate orientation based on sine and cosine weights 
    orient_voxels = np.arctan2(sin_beta,cos_beta)
    # convert negative orientation values (range [-180°, 180°]) to positive values (range [0°,360°])
    orient_voxels[orient_voxels<0] = orient_voxels[orient_voxels<0] + (np.pi*2)
    # convert orientation to n-fold symmetry (e.g. into range [0°,60°])
    orient_voxels_sym = orient_voxels/sym   
    
    # amplitude of n-fold modulation
    amplitude_voxels = np.sqrt(np.square(sin_beta) + np.square(cos_beta))
    
    # calculate mean orientation across voxels weighted by voxels' amplitude
    # (see GridCAT toolbox, Stangl et al., 2017)
    # transformation from polar to cartesian cordinate system 
    x,y = pol2cart(orient_voxels, amplitude_voxels)           
    # find mean across X and across Y separately
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)            
    # transformation back from cartesian to polar cordinate system
    orient_mean, amplitude_mean = cart2pol(mean_x, mean_y)           
    # transform mean grid orientation from range [-pi pi] to [0 2*pi]
    if orient_mean<0: orient_mean = orient_mean + (np.pi*2) 
    # convert orientation to n-fold symmetry (e.g. into range [0°,60°])               
    orient_mean = orient_mean/sym
    
    return orient_mean, orient_voxels_sym, amplitude_voxels 


#%% RUN FIRST-LEVEL GLM

for sub in valspa.subs:
    
    print(f'Run first-level GLM for subject {sub}')
    
    # create first-level directory for sub
    if not os.path.exists(os.path.join(model_path,f'sub-{sub}')): os.makedirs(os.path.join(model_path,f'sub-{sub}'))
    
    # anatomical whole-brain mask for subject with functional resolution
    mask_anat = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_func = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space=space_analysis, suffix='mask', extension='.nii.gz', return_type='file')[0] 
    mask_wholebrain = nilearn.image.resample_to_img(mask_anat, mask_func, interpolation='nearest')
    masks_subs.append(mask_wholebrain)
    
    # ROI mask for estimation of grid orientation 
    mask_roi_file = os.path.join(valspa.derivatives_path, 'masks', f'sub-{sub}', 'func', f'sub-{sub}_space-T1w_desc-{roi}_mask.nii.gz')
    mask_roi      = nilearn.masking.intersect_masks([mask_wholebrain, mask_roi_file], threshold=1, connected=False)
    
    
    # loop through symmetries:  
    for sym in symmetries:
        
        
        """
        RUN ESTIMATION GLM (GLM1) FOR N-FOLD SYMMETRY FOR EACH RUN               
        """
        
        # change names of sin and cos regressors to given symmetry
        regressors_estimation_sym = [reg.replace('n_fold', f'{sym}_fold') for reg in regressors_estimation]
        
        for run in task_info.runs:
            
            # events for run
            events_run = pd.read_csv(layout.get(subject=sub, task=task, run=run, suffix='events')[0].path, sep='\t')
            # filter by events / regressors of interest for GLM 
            events_run = events_run[events_run['trial_type'].str.startswith(tuple(regressors_estimation_sym))] 
            
            # prefix for contrast filenames 
            contrast_info[f'sym-{sym}']['estimation_glm']['contrast_prefix'] = os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}_sym-{sym}_dataset-estimation')
            
            # run glm for run
            fmri.run_first_level_glm(bids_layout = layout, 
                                     sub = sub,
                                     task = task, 
                                     runs = run, 
                                     space_analysis = space_analysis, 
                                     events = events_run, 
                                     mask = mask_wholebrain, 
                                     smoothing = fmri.smoothing, 
                                     contrast_info = contrast_info[f'sym-{sym}']['estimation_glm'])
            
        
        """
        START CROSS-VALIDATION PROCEDURE: ESTIMATE GRID ORIENTATION USING ALL 
        BUT ONE OF THE RUNS AND TEST FOR N-FOLD MODULATION IN LEFT-OUT TEST RUN        
        """
        
        for run in task_info.runs:
            
            # define estimation and test runs
            run_test        = copy.deepcopy(run)
            runs_estimation = np.setdiff1d(task_info.runs, run)
            
            
            """
            ESTIMATE GRID ORIENTATION USING SIN AND COS REGRESSORS OF ESTIMATION GLM          
            """
            
            # get sin and cos betas of ROI voxels and average over estimation runs
            sin_beta = np.array([nilearn.masking.apply_mask(os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run_est}_space-{space_analysis}_sym-{sym}_dataset-estimation_contrast-traject_choice_sin_{sym}_fold_effect_size.nii.gz'), mask_roi) 
                                 for run_est in runs_estimation])
            sin_beta = np.mean(sin_beta, axis=0)
            cos_beta = np.array([nilearn.masking.apply_mask(os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run_est}_space-{space_analysis}_sym-{sym}_dataset-estimation_contrast-traject_choice_cos_{sym}_fold_effect_size.nii.gz'), mask_roi) 
                                 for run_est in runs_estimation])
            cos_beta = np.mean(cos_beta, axis=0)
            
            # estimate mean orientation of ROI
            roi_mean_orientation = estimate_orientation(sin_beta, cos_beta, sym)[0]
            
            
            """
            TEST FOR N-FOLD MODULATION ALIGNED WITH ESTIMATED ORIENTATION IN 
            LEFT-OUT TEST RUN: GLM WITH PARAMETRIC REGRESSOR (MAIN GLM)
            """
            
            # events for run
            events_run = pd.read_csv(layout.get(subject=sub, task=task, run=run_test, suffix='events')[0].path, sep='\t')
            
            # create parametric regressor reflecting n-fold modulation of signal 
            # aligned with estimated mean orientation (cos(sym*(direction_trial-orientation))): 
            # hypothesis is higher signal for aligned vs. misaligned directions/trajectories                                      
            events_run_glm_parametric = copy.deepcopy(events_run)            
            # create parametric regressor
            for direct in task_info.angles:                       
                modulation_value = np.cos(sym*(np.deg2rad(direct) - roi_mean_orientation))                       
                events_run_glm_parametric.loc[events_run_glm_parametric.trial_type==f'traject_angle_{direct}_all_tps', 'modulation'] = copy.deepcopy(modulation_value) 
                events_run_glm_parametric.trial_type = events_run_glm_parametric.trial_type.replace(f'traject_angle_{direct}_all_tps', f'parametric_regressor_{sym}_fold') 
            
            # filter by events / regressors of interest for GLM 
            events_run_glm_parametric = events_run_glm_parametric[events_run_glm_parametric['trial_type'].str.startswith(tuple(regressors_test_glm_parametric))]   
            
            # prefix for contrast filenames 
            contrast_info[f'sym-{sym}']['test_glm_parametric']['contrast_prefix'] = os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}_sym-{sym}_dataset-test_glm-parametric')
            
            # run glm for run
            fmri.run_first_level_glm(bids_layout = layout, 
                                     sub = sub,
                                     task = task, 
                                     runs = run_test, 
                                     space_analysis = space_analysis, 
                                     events = events_run_glm_parametric, 
                                     mask = mask_wholebrain, 
                                     smoothing = fmri.smoothing, 
                                     contrast_info = contrast_info[f'sym-{sym}']['test_glm_parametric'])
            
            
            """
            TEST FOR N-FOLD MODULATION ALIGNED WITH ESTIMATED ORIENTATION IN 
            LEFT-OUT TEST RUN: GLM WITH SEPARATE REGRESSORS FOR DIRECTION BINS
            (FOR VISUALIZATION OF 6-FOLD EFFECT)
            """
            
            if sym==6:
                
                # hexadirectional modulation means sinusoidal pattern with 6 peaks
                # (aligned) and 6 troughs (misaligned): create regressors for bins 
                # of directions surrounding these peaks/troughs +/-15° 
                # (result: 12 regressors, each reflecting a 30° bin)
                
                # predicted peaks/troughs of aligned and misaligned directions 
                pred_dir_peaks_aligned = np.arange(np.rad2deg(roi_mean_orientation), np.rad2deg(roi_mean_orientation)+360, 360/sym)
                if len(pred_dir_peaks_aligned)>sym:
                    pred_dir_peaks_aligned = pred_dir_peaks_aligned[:-1]                        
                pred_dir_peaks_misaligned = pred_dir_peaks_aligned + 360/sym/2
                pred_dir_peaks_aligned[np.where(pred_dir_peaks_aligned>360)] = pred_dir_peaks_aligned[np.where(pred_dir_peaks_aligned>360)] - 360
                pred_dir_peaks_misaligned[np.where(pred_dir_peaks_misaligned>360)] = pred_dir_peaks_misaligned[np.where(pred_dir_peaks_misaligned>360)] - 360
                
                # predicted bins of aligned and misaligned directions   
                pred_dir_bins_aligned = [task_info.angles[abs((task_info.angles - peak + 180) % 360 - 180) <= 15] for peak in pred_dir_peaks_aligned]
                pred_dir_bins_misaligned = [task_info.angles[abs((task_info.angles - peak + 180) % 360 - 180) <= 15] for peak in pred_dir_peaks_misaligned]
                
                # create regressors                 
                events_run_glm_dir_bins = copy.deepcopy(events_run)  

                for i_dir_list, dir_list in enumerate(pred_dir_bins_aligned):
                    for direct in dir_list:
                        events_run_glm_dir_bins.trial_type = events_run_glm_dir_bins.trial_type.replace(f'traject_angle_{direct}_all_tps', f'traject_direction_bin_{hypo_dir_peaks_aligned[i_dir_list]}') 
                
                for i_dir_list, dir_list in enumerate(pred_dir_bins_misaligned):
                    for direct in dir_list:
                        events_run_glm_dir_bins.trial_type = events_run_glm_dir_bins.trial_type.replace(f'traject_angle_{direct}_all_tps', f'traject_direction_bin_{hypo_dir_peaks_misaligned[i_dir_list]}') 
                
                # filter by events / regressors of interest for GLM 
                events_run_glm_dir_bins = events_run_glm_dir_bins[events_run_glm_dir_bins['trial_type'].str.startswith(tuple(regressors_test_glm_dir_bins))] 
                
                # prefix for contrast filenames 
                contrast_info[f'sym-{sym}']['test_glm_dir_bins']['contrast_prefix'] = os.path.join(model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}_sym-{sym}_dataset-test_glm-direction_bins')
                
                # run glm for run
                fmri.run_first_level_glm(bids_layout = layout, 
                                         sub = sub,
                                         task = task, 
                                         runs = run_test, 
                                         space_analysis = space_analysis, 
                                         events = events_run_glm_dir_bins, 
                                         mask = mask_wholebrain, 
                                         smoothing = fmri.smoothing, 
                                         contrast_info = contrast_info[f'sym-{sym}']['test_glm_dir_bins'])
                
                
        """
        AVERAGE RESULT MAPS OVER CROSS-VALIDATION FOLDS AND REGISTER TO MNI SPACE
        """
        
        # images to average over 
        imgs_avg_cv = [
            f'sub-{sub}_task-{task}_run-X_space-{space_analysis}_sym-{sym}_dataset-test_glm-parametric_contrast-parametric_regressor_{sym}_fold_effect_size.nii.gz'
            ]
        
        if sym==6:            
            for peak_bin in hypo_dir_peaks_aligned:
                imgs_avg_cv.append(f'sub-{sub}_task-{task}_run-X_space-{space_analysis}_sym-{sym}_dataset-test_glm-direction_bins_contrast-direction_bin_{peak_bin}_effect_size.nii.gz')            
            for peak_bin in hypo_dir_peaks_misaligned:
                imgs_avg_cv.append(f'sub-{sub}_task-{task}_run-X_space-{space_analysis}_sym-{sym}_dataset-test_glm-direction_bins_contrast-direction_bin_{peak_bin}_effect_size.nii.gz')
        
        for img in imgs_avg_cv:
            
            # list of all images of cv-folds
            img_list = [os.path.join(model_path, f'sub-{sub}', img.replace('run-X', f'run-{run}')) for run in task_info.runs]
            
            # mean across cv-folds
            mean_img = nilearn.image.mean_img(img_list)
            mean_img.to_filename(os.path.join(model_path, f'sub-{sub}', img.replace('run-X', f'run-all')))
                        
            # register image in subject T1 space to MNI standard space using ANTS
            # requires a reference image (MNI in functional space) and a transformation matrix  
            t1_nii_file  = os.path.join(model_path, f'sub-{sub}', img.replace('run-X', f'run-all'))
            mni_nii_file = t1_nii_file.replace(f'space-{space_analysis}', f'space-{space_transform}')
            reference    = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space='MNI152NLin2009cAsym', suffix='mask', extension='.nii.gz', return_type='file')[0]
            transform    = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', extension='.h5', return_type='file')[3]             
            os.system(f'ANTSENV antsApplyTransforms -i {t1_nii_file} -o {mni_nii_file} -r {reference} -t {transform}')
            
            
#%% RUN SECOND-LEVEL MODEL USING FSL RANDOMISE

# second-level contrasts of interest
sl_contrasts = [f'sym-{sym}_dataset-test_glm-parametric_contrast-parametric_regressor_{sym}_fold' for sym in symmetries]

# masks for second-level analysis
sl_masks = {'wholebrain': f'wholebrain_space-{space_analysis}-to-{space_transform}',
            'EC': 'EC_SL_Freesurfer_Juelich' # small volume correction mask for entorhinal cortex 
            }

# masks for second-level analysis: create intersection of all subjects masks
# so that analysis is performend only in voxels shared across subjects 
if not os.path.exists(os.path.join(valspa.second_level_path, task, 'masks')): os.makedirs(os.path.join(valspa.second_level_path, task, 'masks'))
sl_wholebrain_mask = os.path.join(valspa.second_level_path, task, 'masks', f'wholebrain_space-{space_analysis}-to-{space_transform}.nii.gz')
if not os.path.isfile(sl_wholebrain_mask):
    # merge all subject-specific images in MNI space for a given contrast and 
    # mask voxels shared across subjects (same for any contrast)
    sub_con_imgs = []        
    for sub in valspa.subs:       
        con_img = os.path.join(valspa.first_level_path, task, model_name, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_transform}_{sl_contrasts[0]}_effect_size.nii.gz')
        sub_con_imgs.append(con_img)
    # concatenate images
    sub_con_imgs_concat = nilearn.image.concat_imgs(sub_con_imgs)
    # mask by shared voxels
    mask_img = nilearn.image.math_img('img.all(axis=3)', img=sub_con_imgs_concat)
    mask_img.to_filename(sl_wholebrain_mask)   
    
# intersect small volume correction mask for entorhinal cortex with this wholebrain mask
mask_img = nilearn.masking.intersect_masks([sl_wholebrain_mask, os.path.join(valspa.derivatives_path, 'masks', 'mni', 'EC_SL_Freesurfer_Juelich')], threshold=1, connected=False)
mask_img.to_filename(os.path.join(valspa.second_level_path, task, 'masks', 'EC_SL_Freesurfer_Juelich.nii.gz')) 
    
# run randomise
fmri.run_second_level_randomise(valspa = valspa, 
                                subjects = valspa.subs, 
                                task = task, 
                                model_name = model_name, 
                                second_level_masks = sl_masks,
                                second_level_contrasts = sl_contrasts)


#%% PLOT SECOND-LEVEL RESULTS WHOLEBRAIN MAP FOR HEXADIRECTIONAL EFFECT 
# plot wholebrain t-map thresholded at p(uncorr) < .01, with significant entorhinal
# cluster as black outline

fig_contrast = 'sym-6_dataset-test_glm-parametric_contrast-parametric_regressor_6_fold'
sl_model_path = os.path.join(valspa.second_level_path, task, model_name)

# create mask of p(uncorr) < .01 and mask the t-map accordingly 
tmap_file       = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{space_transform}_tstat1.nii.gz')
pos_voxp_file   = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{space_transform}_vox_p_tstat1.nii.gz')
pos_voxp_thr    = nilearn.image.math_img('img>=0.99', img=pos_voxp_file)
tmap_img_masked = nilearn.image.math_img('img1 * img2', img1=tmap_file, img2=pos_voxp_thr)
tmap_img_masked.to_filename(os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{space_transform}_tstat1_masked_positive_uncorr_01.nii.gz')) 

# create mask of small volume corrected EC TFCE image for pFWE < .05
smc_tfce_file   = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_EC_space-{space_transform}_tfce_corrp_tstat1.nii.gz')
smc_tfce_mask   = nilearn.image.math_img('img>=0.95', img=smc_tfce_file) 
smc_tfce_mask.to_filename(os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_EC_space-{space_transform}_tfce_corrp_tstat1_masked_corr_05.nii.gz')) 
smc_tfce_mask.to_filename(os.path.join(valspa.derivatives_path, 'masks', 'mni', 'hexadirectional_effect_EC.nii.gz'))

fig_file = os.path.join(sl_model_path, f'contrast-{fig_contrast}', 'figure-wholebrain-map-hexadirectional-effect.pdf')
display = nilearn.plotting.plot_stat_map(tmap_img_masked, 
                                         bg_img = os.path.join(valspa.derivatives_path, 'masks', 'mni', 'MNI152_T1_1mm_brain.nii.gz'), 
                                         black_bg = False, 
                                         cut_coords = (20,-6,-36), 
                                         draw_cross = False)
display.add_contours(smc_tfce_mask, linewidths=1.5, levels=[0.5], colors='black')
display.savefig(fig_file)


#%% CLUSTER INFORMATION SECOND-LEVEL RESULTS WHOLEBRAIN MAP

# create mask of p(uncorr) < .001 and mask the t-map accordingly 
tmap_file       = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{space_transform}_tstat1.nii.gz')
pos_voxp_file   = os.path.join(sl_model_path, f'contrast-{fig_contrast}', f'positive_wholebrain_space-{space_transform}_vox_p_tstat1.nii.gz')
pos_voxp_thr    = nilearn.image.math_img('img>=0.999', img=pos_voxp_file)
tmap_img_masked = nilearn.image.math_img('img1 * img2', img1=tmap_file, img2=pos_voxp_thr)

# get cluster information using nilearn 
clusters_table = nilearn.reporting.get_clusters_table(stat_img = tmap_img_masked, 
                                                      stat_threshold = 0.001,
                                                      two_sided = False,
                                                      min_distance = 0)

# add atlas labels using FSL atlasquery
fmri.get_second_level_cluster_labels(clusters_table = clusters_table, 
                                     output_path = os.path.join(sl_model_path, f'contrast-{fig_contrast}'))




