#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# MODULE FOR FMRI ANALYSES
#
# This module defines variables and functions used for fMRI analyses.
#
# fMRI analyses use data in BIDS format and preprocessed with fMRIPrep.
# fMRI analyses, especially first and second level modeling, are based on 
# nilearn and FSL, see resources:
# - https://nilearn.github.io/stable/index.html
# - https://lukas-snoek.com/NI-edu/section_intros/7_nilearn.html 
# - https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Randomise/UserGuide
# 
# AUTHOR: Alexander Nitsch
# CONTACT: nitsch@cbs.mpg.de
# Max Planck Institute for Human Cognitive and Brain Sciences
# DATE: 2022
#
# =============================================================================
"""

#%% IMPORT PACKAGES

import numpy as np
import pandas as pd
import copy
import os
import os.path

import nibabel as nib
import nilearn
from nilearn.glm.first_level import FirstLevelModel
import nilearn.plotting
from statsmodels.stats.outliers_influence import variance_inflation_factor

import subprocess


#%% VARIABLES

tr = 1.5
slice_time_reference = 0.5
smoothing = 6
space_standard = 'MNI152NLin2009cAsym'
space_subject = 'T1w'

# task-related events of no interest 
regressors_no_interest = ['left_button', 'right_button', 'stop_scanner_screen_start'] 

# confound regressors for GLMs (confounds estimated by fMRIPrep) 
# 1) 24 motion parameters (original 6 + temporal derivatives and quadratic terms)
# 2) framewise displacement
# 3) 12 global signal regressors (CSF, WM, global signal + temporal derivatives and quadratic terms)
# 4) cosine regressors to account for temporal low-frequency signal drifts
confound_regressors = [
    'trans',
    'rot', 
    'framewise_displacement',
    'white_matter',
    'csf',
    'global_signal',
    'cosine'
    ]

# number of permutations for FSL randomise 
n_second_level_randomise_permutations = 10000


#%% FUNCTIONS


def preprocess_confounds(confounds, confound_regressors):
    """
    Preprocess confound regressors

    Parameters
    ----------
    confounds : confounds df (by fMRIPrep)
    confound_regressors : confound regressors to be included in GLM

    Returns
    -------
    confounds_filtered : preprocessed confounds

    """

    # filter confounds to include only selected confound regressors
    confound_reg_names = [col for col in confounds.columns if
                          any([conf in col for conf in confound_regressors])]  
    confound_reg_names.remove('csf_wm') # remove as csf and wm are already included individually          
    confounds_filtered = confounds[confound_reg_names]
    # replace NaNs in confound columns by mean value
    for conf in confounds_filtered:
        conf_mean_value = confounds_filtered[conf].mean()
        confounds_filtered[conf][confounds_filtered[conf].isnull()] = conf_mean_value        
    return confounds_filtered


def demean_regressors(events):
    """
    Check whether regressors in events are parametrically modulated 
    and demean them if this is the case

    Parameters
    ----------
    events : events df

    Returns
    -------
    events : events df with demeaned regressors

    """

    regressors = events['trial_type'].unique().tolist()
    for reg in regressors:
        mod = events.loc[events['trial_type']==reg, 'modulation']
        # only demean if parametrically modulated (not all modulation values are 1)
        if not mod.eq(1).all():
            mod_demeaned = mod - mod.mean()
            events.loc[events['trial_type']==reg, 'modulation'] = mod_demeaned[:]
    return events


def run_first_level_glm(bids_layout, sub, task, runs, space_analysis, events, mask, smoothing, contrast_info):
    """
    Run a first level GLM for a given subject, either for all runs at once
    or for a given run only. 

    Parameters
    ----------
    bids_layout : layout of BIDS dataset (data structure) 
    sub : subject
    task : task
    runs : runs to be modeled  
    space_analysis : space of fMRI data (standard or native space)
    events : events to create regressors for model. events should only include
    events of interest (filtered).
    mask : mask of included voxels
    smoothing : amount of smoothing 
    contrast_info : dict of contrast definitions and prefix for output filenames 

    Returns
    -------
    Saves contrast nii files.

    """
    
    if not isinstance(runs, list):
        runs = [runs]
    
    if not isinstance(events, list):
        events = [events]
    
    # initialize lists to collect information for all runs
    run_imgs, confounds = [], []
   
    for i_run, run in enumerate(runs):
        
        # fMRI run file
        run_file = bids_layout.get(subject=sub, task=task, run=run, space=space_analysis, scope='fMRIPrep', suffix='bold', extension='nii.gz', return_type='file')[0] 
        run_imgs.append(run_file)
        
        # events for run: events were passed to the function and include only
        # those of interest for the model 
        # demean any parametric regressors in events 
        events[i_run] = demean_regressors(events[i_run])        
        
        # confounds for run
        confounds_run = pd.read_csv(bids_layout.get(subject=sub, task=task, run=run, scope='fMRIPrep', extension='.tsv')[0].path, sep='\t')
        confounds_run = preprocess_confounds(confounds_run, confound_regressors)
        confounds.append(confounds_run)
    
    
    # initialize FirstLevelModel class 
    flm = FirstLevelModel(
        t_r             = tr,
        slice_time_ref  = slice_time_reference,
        hrf_model       = 'glover',
        drift_model     = None, # cosine regressors included in confounds
        mask_img        = mask,
        smoothing_fwhm  = smoothing,
        noise_model     = 'ar1', # accounting for temporal autocorrelation
        n_jobs          = 1,
        minimize_memory = False, 
        verbose         = True            
        )
    
    # fit GLM 
    flm.fit(run_imgs=run_imgs, events=events, confounds=confounds)
    
    # # check design matrix: correlations between regressors and variance inflation factor
    # flm_dm = {}
    # for i_run, run in enumerate(runs):
    #     flm_dm[f'run-{run}'] = {}
    #     flm_dm[f'run-{run}']['dm_corr'] = flm.design_matrices_[i_run].corr()
    #     flm_dm[f'run-{run}']['dm_vif'] = pd.DataFrame()
    #     flm_dm[f'run-{run}']['dm_vif']['vif_factor'] = [variance_inflation_factor(flm.design_matrices_[i_run].values, i) for i in range(flm.design_matrices_[i_run].shape[1])]
    #     flm_dm[f'run-{run}']['dm_vif']['regressor'] = copy.deepcopy(flm.design_matrices_[i_run].columns)
    
    # compute contrasts and save all respective statistics 
    # if multiple runs are modeled, contrasts are computed as fixed effects across runs
    contrast_imgs = {}
    for con in contrast_info['contrast_definitions']:
        contrast_imgs[con] = flm.compute_contrast(contrast_info['contrast_definitions'][con], stat_type='t', output_type='all')    
        for output in contrast_imgs[con]:
            contrast_imgs[con][output].to_filename(contrast_info['contrast_prefix'] + f'_contrast-{con}_{output}.nii.gz')
        
    # return events, confounds, flm_dm


def run_second_level_randomise(valspa, subjects, task, model_name, second_level_masks, second_level_contrasts):
    """
    Run second level analysis using FSL randomise. Use one-sample permutation test
    with TFCE for correction. 

    Parameters
    ----------
    valspa : valspa class with project-related info (e.g. paths)
    subjects : subjects to be included in second level
    task : task
    model_name : model name of GLM / analysis
    second_level_masks : masks, e.g. wholebrain or small volume correction 
    (dict with mask filenames)
    second_level_contrasts : contrasts of interest

    Returns
    -------
    Saves second level results nii files.

    """
    
    # second-level model path
    second_level_model_path = os.path.join(valspa.second_level_path, task, model_name)
    if not os.path.exists(second_level_model_path): os.makedirs(second_level_model_path)
    
    # run second-level analysis for each contrast
    for con in second_level_contrasts:
        
        # define contrast string for folder names
        if not con.startswith('contrast'):
            con_string = f'contrast-{con}'
        else:
            con_string = f'{con}'
        
        
        # step 1: merge all subject-specific contrast images (3D) into 4D image and 
        # create negative version for two-sided testing
        
        # create 4D directory
        if not os.path.exists(os.path.join(second_level_model_path, '4D_images')): os.makedirs(os.path.join(second_level_model_path, '4D_images'))
        
        # create a positive (original) 4D image of the contrast and also a negative 
        # version of the image for two-sided testing (as positive & negative effects 
        # need to be tested separately by randomise) 
        out_4D_pos = os.path.join(second_level_model_path, '4D_images', f'{con_string}_positive.nii.gz')
        out_4D_neg = os.path.join(second_level_model_path, '4D_images', f'{con_string}_negative.nii.gz')
        
        if not os.path.isfile(out_4D_pos):
            sub_con_imgs = []        
            for sub in subjects:       
                con_img = os.path.join(valspa.first_level_path, task, model_name, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_standard}_{con}_effect_size.nii.gz')
                sub_con_imgs.append(con_img)
            
            # concatenate images
            sub_con_imgs_concat = nilearn.image.concat_imgs(sub_con_imgs)
            # save positive 4D image
            sub_con_imgs_concat.to_filename(out_4D_pos)
            # save negative 4D image
            sub_con_imgs_concat_neg = nilearn.image.math_img('-img', img=sub_con_imgs_concat)
            sub_con_imgs_concat_neg.to_filename(out_4D_neg)
        
        
        # step 2: permutation test using randomise (case: one-sample test, use
        # TFCE but also output uncorrected and cluster results)
        
        # create output directory
        out_dir = os.path.join(second_level_model_path, con_string)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        
        for mask in second_level_masks:
            
            mask_file = os.path.join(valspa.second_level_path, task, 'masks', f'{second_level_masks[mask]}.nii.gz')
        
            output = os.path.join(out_dir, f'positive_{mask}_space-{space_standard}')    
            randomise_cmd = f'FSL --cluster randomise -i {out_4D_pos} -o {output} -1 -T -m {mask_file} -x --uncorrp -n {n_second_level_randomise_permutations} -C 0.001'
            print(randomise_cmd) 
            # os.system(randomise_cmd)    
            
            output = os.path.join(out_dir, f'negative_{mask}_space-{space_standard}')    
            randomise_cmd = f'FSL --cluster randomise -i {out_4D_neg} -o {output} -1 -T -m {mask_file} -x --uncorrp -n {n_second_level_randomise_permutations} -C 0.001'
            print(randomise_cmd) 
            # os.system(randomise_cmd)


def get_second_level_cluster_labels(clusters_table, output_path):
    """
    Get atlas labels for clusters at second level. Clusters were already identified
    using nilearn. This function mainly adds atlas labels using FSL atlasquery.

    Parameters
    ----------
    cluster_table : table with clusters (nilearn.reporting.get_clusters_table output)
    output_path : output path to save csv file with table

    Returns
    -------
    Saves csv file.

    """
    
    # convert cluster extent of mm3 to number of voxels
    clusters_table['Cluster Size (mm3)'] = pd.to_numeric(clusters_table['Cluster Size (mm3)'])
    clusters_table['Cluster Size (n voxels)'] = clusters_table['Cluster Size (mm3)'] / 2.5**3
    
    # round to integers
    clusters_table['Peak Stat'] = pd.to_numeric(clusters_table['Peak Stat'])
    for col in clusters_table.columns[1:7]:
        if col == 'Peak Stat':
            clusters_table[col] = np.round(clusters_table[col],2)
        else:
            clusters_table[col] = np.round(clusters_table[col])
    
    # get atlas labels for clusters using FSL atlasquery
    atlas_list = ['Harvard-Oxford Cortical Structural Atlas',
                  'Harvard-Oxford Subcortical Structural Atlas',
                  'Juelich Histological Atlas']
    
    clusters_table['Atlas label'] = np.nan
    
    for i_row in range(len(clusters_table)):
        # coordinates
        x_coord = clusters_table.loc[i_row, 'X'].item() 
        y_coord = clusters_table.loc[i_row, 'Y'].item() 
        z_coord = clusters_table.loc[i_row, 'Z'].item() 
        
        label_string = ''
        for a in atlas_list:
            label_string = label_string + str(subprocess.check_output(f"FSL --cluster atlasquery -a \"{a}\" -c {x_coord},{y_coord},{z_coord} | sed 's/<b>//g' | sed 's/<\/b><br>/\,/g'", shell=True))
        
        clusters_table.loc[i_row, 'Atlas label'] = copy.deepcopy(label_string)
    
    # save as csv file
    clusters_info_file = os.path.join(output_path, 'clusters_info.tsv')
    clusters_table.to_csv(clusters_info_file, sep='\t')
    
    
    
    
    
    
    
    
    
