#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# HEXADIRECTIONAL fMRI ANALYSIS
# (ANALYSIS OF GRID-LIKE REPRESENTATION)
#
# Extract results data and run control analyses regarding grid orientations. 
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

import nibabel as nib
import nilearn
import nilearn.masking

import pycircstat
import astropy.stats


#%% SET VARIABLES

# bids data layout
layout      = BIDSLayout(valspa.bids_path, derivatives=True)

# task-relevant variables 
task        = 'DEC'
task_info   = eval(f'valspa.{task.lower()}()')

# GLM model name
model_name  = 'hexadirectional_analysis'
fl_model_path  = os.path.join(valspa.first_level_path, task, model_name)
hexa_path = os.path.join(valspa.derivatives_path, 'hexadirectional_analysis')
plot_group_path = os.path.join(hexa_path, 'group')

# space for analysis (MNI or subject native space)
space_analysis  = copy.deepcopy(valspa.fmri.space_subject)
space_transform = copy.deepcopy(valspa.fmri.space_standard)

# symmetries (n-fold modulation of fMRI signal): 
# hexadirectional=6-fold symmetry of interest, additionally control symmetries 
symmetries = np.arange(4,9) 

# hypothesized directional peaks of 6-fold effect relative to arbitrary
# grid orientation of 0° 
hypo_dir_peaks_aligned = task_info.angles[::6]
hypo_dir_peaks_misaligned = hypo_dir_peaks_aligned + 30
hypo_dir_peaks = task_info.angles[::3]

# ROIs 
roi_fl = 'EC_Freesurfer' # subject-specific EC masks as used in first level to estimate grid orientation
roi_sl = 'hexadirectional_effect_EC' # significant EC cluster in second level analysis 
mask_file_sig_cluster = os.path.join(valspa.derivatives_path, 'masks', 'mni', f'{roi_sl}.nii.gz')


#%% FUNCTIONS FOR ESTIMATION OF GRID ORIENTATION AND POLAR PLOTS

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


def polar_histogram(voxel_orientations, mean_orientation, rayleigh_test_p, figure_name):
    """
    Plot polar histogram of voxel orientations, with a range of [0°,60°] across 
    the entire circle and with the mean orientation as arrow

    Parameters
    ----------
    voxel_orientations : voxel-wise orientations in 60°-space (radians)
    mean_orientation : mean orientation across voxels in 60°-space (radians)
    rayleigh_test_p : p value of Rayleigh test of uniformity
    figure_name : path for figure output

    """

    # make histogram of orientations in bins, input data in radians with range [0°,60°]
    # must be converted to range [0°,360°] for accurate plotting
    bin_size = 10
    a , b = np.histogram(np.rad2deg(voxel_orientations*6), bins=np.arange(0, 360+bin_size, bin_size))
    centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])        
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    # ticks will be set at these positions but change labels in order to have labels 
    # in range [0°,60°]
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/6))
    ax.set_xticklabels([f'{string}°' for string in np.arange(0,60,5)])
    ax.bar(centers, a, width=np.deg2rad(bin_size), color=valspa.color_petrol, edgecolor='black')
    ax.arrow(mean_orientation*6, 0, 0, np.max(a)-2, width = 0.005, head_width=0.15, head_length=0.1*np.max(a), overhang=1, zorder=2, color='red')
    ax.set_title(f'Polar histogram of voxel orientations (Rayleigh p={np.round(rayleigh_test_p,2)})')
    plt.savefig(figure_name, bbox_inches='tight')
    plt.show()
    

def polar_scatterplot(voxel_orientations, voxel_amplitudes, mean_orientation, figure_name):
    """
    Plot polar histogram of voxel orientations, with a range of [0°,60°] across 
    the entire circle, color-coded as a function of the amplitude of hexadirectional 
    modulation and with the mean orientation as arrow
    
    Parameters
    ----------
    voxel_orientations : voxel-wise orientations in 60°-space (radians)
    voxel_amplitudes : voxel-wise amplitudes
    mean_orientation : mean orientation across voxels in 60°-space (radians)
    figure_name : path for figure output

    """

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='polar')
    # ticks will be set at these positions but change labels in order to have labels 
    # in range [0°,60°]
    ax.set_xticks(np.arange(0,2*np.pi,np.pi/6))
    ax.set_xticklabels([f'{string}°' for string in np.arange(0,60,5)])
    # scatterplot with amplitude as radii and color
    # radians with range [0°,60°] must be converted to range [0°,360°] for accurate plotting
    ax.scatter(voxel_orientations*6, voxel_amplitudes, c=voxel_amplitudes)
    ax.arrow(mean_orientation*6, 0, 0, np.max(voxel_amplitudes)-0.05, width = 0.005, head_width=0.15, head_length=0.05, overhang=1, zorder=2, color='red')
    ax.set_title('Polar scatterplot of voxel orientations as a function of amplitude')
    plt.savefig(figure_name, bbox_inches='tight')
    plt.show()


#%% EXTRACT DATA FOR SIGNIFICANT CLUSTER IN EC

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
    
    # conditions/contrasts of interest as saved in first level folder
    contrast_imgs = [
        'sym-6_dataset-test_glm-parametric_contrast-parametric_regressor_6_fold_effect_size'
        ]
    contrast_imgs.extend([f'sym-6_dataset-test_glm-direction_bins_contrast-direction_bin_{peak_bin}_effect_size' for peak_bin in hypo_dir_peaks])

    # names of conditions for df
    condition_names = [
        '6-fold_glm_parametric_contrast_parametric'
        ]
    condition_names.extend([f'6-fold_glm_peak_bins_contrast_bin_{peak_bin}°' for peak_bin in hypo_dir_peaks])
           
    # extract roi values
    for i_con, con in enumerate(contrast_imgs):
        contrast_file = os.path.join(fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_transform}_{con}.nii.gz') 
        roi_mean = nilearn.masking.apply_mask(contrast_file, mask_file_sig_cluster).mean()
        
        subject.append(sub)
        condition.append(condition_names[i_con])
        effect_size.append(roi_mean)


# create dataframe
data_significant_cluster = pd.DataFrame({'subject': subject,
                                         'condition': condition,
                                         'effect_size': effect_size})


#%% EXTRACT DATA FOR ALL (CONTROL) SYMMETRIES IN EC ROI USED FOR FIRST LEVEL

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
    
    # roi mask as used for first-level    
    mask_file_roi_fl        = os.path.join(valspa.derivatives_path, 'masks', f'sub-{sub}', 'func', f'sub-{sub}_space-T1w_desc-{roi_fl}_mask.nii.gz')
    mask_file_wholebrain_fl = os.path.join(valspa.derivatives_path, 'masks', f'sub-{sub}', 'func', f'sub-{sub}_space-T1w_desc-wholebrain_mask.nii.gz')
    mask_roi_fl             = nilearn.masking.intersect_masks([mask_file_roi_fl, mask_file_wholebrain_fl], threshold=1, connected=False)
    
    for sym in symmetries:
    
        # conditions/contrasts of interest as saved in first level folder
        contrast = f'sym-{sym}_dataset-test_glm-parametric_contrast-parametric_regressor_{sym}_fold_effect_size' 
        contrast_file = os.path.join(fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_analysis}_{contrast}.nii.gz') 
        roi_mean = nilearn.masking.apply_mask(contrast_file, mask_roi_fl).mean()
        
        subject.append(sub)
        condition.append(f'{sym}-fold')
        effect_size.append(roi_mean)


# create dataframe
data_fl_roi = pd.DataFrame({'subject': subject,
                            'condition': condition,
                            'effect_size': effect_size})


#%% CHECK SPATIAL STABILITY OF GRID ORIENTATIONS IN SIGNIFICANT EC CLUSTER
# similarity / clustering of orientations across voxels

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
    
    # create path for figures
    if not os.path.exists(os.path.join(hexa_path, f'sub-{sub}')): os.makedirs(os.path.join(hexa_path, f'sub-{sub}'))
    
    # average sin & cos regressor images of estimation GLM (GLM1) over all runs (to increase power)
    sin_img_t1_file = os.path.join(fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_analysis}_sym-6_dataset-estimation_contrast-traject_choice_sin_6_fold_effect_size.nii.gz')
    sin_img_t1_all_runs = nilearn.image.mean_img([sin_img_t1_file.replace('run-all', f'run-{run}') for run in task_info.runs])
    sin_img_t1_all_runs.to_filename(sin_img_t1_file)
    cos_img_t1_file = os.path.join(fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_analysis}_sym-6_dataset-estimation_contrast-traject_choice_cos_6_fold_effect_size.nii.gz')
    cos_img_t1_all_runs = nilearn.image.mean_img([cos_img_t1_file.replace('run-all', f'run-{run}') for run in task_info.runs])
    cos_img_t1_all_runs.to_filename(cos_img_t1_file)
      
    # transform them from T1 to MNI space 
    sin_img_mni_file = sin_img_t1_file.replace(f'space-{space_analysis}', f'space-{space_transform}')
    cos_img_mni_file = cos_img_t1_file.replace(f'space-{space_analysis}', f'space-{space_transform}')
    if not os.path.isfile(sin_img_mni_file):
        reference = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space='MNI152NLin2009cAsym', suffix='mask', extension='.nii.gz', return_type='file')[0]
        transform = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', extension='.h5', return_type='file')[3] 
        os.system(f'ANTSENV antsApplyTransforms -i {sin_img_t1_file} -o {sin_img_mni_file} -r {reference} -t {transform}')
        os.system(f'ANTSENV antsApplyTransforms -i {cos_img_t1_file} -o {cos_img_mni_file} -r {reference} -t {transform}')
    
    # get sin and cos betas 
    sin_beta = nilearn.masking.apply_mask(sin_img_mni_file, mask_file_sig_cluster)
    cos_beta = nilearn.masking.apply_mask(cos_img_mni_file, mask_file_sig_cluster)
    
    # estimate orientations
    orient_mean, orient_voxels_sym, amplitude_voxels = estimate_orientation(sin_beta, cos_beta, sym=6)
    
    # Rayleigh test of uniformity: test the alternative hypothesis that voxel 
    # orientations are not uniformly distributed (requires transformation of
    # angles to range of [0°,360°])
    rayleigh_p, rayleigh_z = pycircstat.tests.rayleigh(orient_voxels_sym*6)
    
    # amgular difference between mean orientation and 45° (reference direction / diagonal of value space)
    orient_mean_diff_to_45 = np.rad2deg(orient_mean) - 45
    # ori_diff = abs((ori_diff + 180) % 360 - 180) # difference in 360°-space
    orient_mean_diff_to_45 = abs((orient_mean_diff_to_45 + 30) % 60 - 30) # difference in 60°-space

    # # plot polar histogram of voxel orientations with mean orientation as arrow
    # polar_histogram(voxel_orientations = orient_voxels_sym, 
    #                 mean_orientation = orient_mean,
    #                 rayleigh_test_p = rayleigh_p,
    #                 figure_name = os.path.join(hexa_path, f'sub-{sub}', f'sub-{sub}_roi-{roi_sl}_figure-voxel-orientations-histogram-with-mean-orientation-weighted-by-amplitude.pdf'))
    
    # # plot polar scatterplot of voxel orientations with voxels colored by amplitude
    # polar_scatterplot(voxel_orientations = orient_voxels_sym,
    #                   voxel_amplitudes = amplitude_voxels,
    #                   mean_orientation = orient_mean,
    #                   figure_name = os.path.join(hexa_path, f'sub-{sub}', f'sub-{sub}_roi-{roi_sl}_figure-voxel-orientations-scatterplot-with-mean-orientation-weighted-by-amplitude.pdf'))
    
    
    # save data
    name_conditions_variables = {'mean_orientation_amplitude': orient_mean,
                                 'mean_orientation_amplitude_deg': np.rad2deg(orient_mean),
                                 'rayleigh_p': rayleigh_p,
                                 'rayleigh_z': rayleigh_z,
                                 'mean_orientation_amplitude_diff_to_45_deg': orient_mean_diff_to_45} 
    
    for c in name_conditions_variables:
        subject.append(sub)
        condition.append(c)
        effect_size.append(name_conditions_variables[c])


# create dataframe
data_orient_spatial_stab = pd.DataFrame({'subject': subject,
                                         'condition': condition,
                                         'effect_size': effect_size})


# clustering of orientations across subjects
circ_mean = np.rad2deg(scipy.stats.circmean(data_orient_spatial_stab[data_orient_spatial_stab.condition=='mean_orientation_amplitude'].effect_size*6)/6)
circ_std = np.rad2deg(scipy.stats.circstd(data_orient_spatial_stab[data_orient_spatial_stab.condition=='mean_orientation_amplitude'].effect_size*6)/6)
# check whether they cluster around 45°
v_test_p = astropy.stats.circstats.vtest(data_orient_spatial_stab[data_orient_spatial_stab.condition=='mean_orientation_amplitude'].effect_size*6, mu=np.deg2rad(45)*6)


#%% CHECK TEMPORAL STABILITY OF GRID ORIENTATIONS IN SIGNIFICANT EC CLUSTER
# similarity of voxel orientations across runs (same cross-validation procedure
# as for first level model: similarity between estimation runs and test run)

# criterion to classify voxel as stable: if orientations differ by max. 15°
criterion_stable_voxel = 15 

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []

# number of voxels in significant cluster
n_voxels_sig_cluster = np.sum(nib.load(mask_file_sig_cluster).get_fdata()==1)


for sub in valspa.subs:
    
    # create path for figures
    if not os.path.exists(os.path.join(hexa_path, f'sub-{sub}')): os.makedirs(os.path.join(hexa_path, f'sub-{sub}'))
    
    # initialize array with orientation difference in each cross-validation fold
    voxels_orient_diff = np.empty((n_voxels_sig_cluster, task_info.n_runs))
    
    # get sin and cos betas of estimation GLM (GLM1) of all individual runs 
    sin_runs = np.empty((n_voxels_sig_cluster, task_info.n_runs)) 
    cos_runs = np.empty((n_voxels_sig_cluster, task_info.n_runs)) 
    
    for i_run, run in enumerate(task_info.runs): 
        sin_img_t1_file = os.path.join(fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}_sym-6_dataset-estimation_contrast-traject_choice_sin_6_fold_effect_size.nii.gz')
        sin_img_mni_file = sin_img_t1_file.replace(f'space-{space_analysis}', f'space-{space_transform}')
        cos_img_t1_file = os.path.join(fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-{run}_space-{space_analysis}_sym-6_dataset-estimation_contrast-traject_choice_cos_6_fold_effect_size.nii.gz')
        cos_img_mni_file = cos_img_t1_file.replace(f'space-{space_analysis}', f'space-{space_transform}')
        
        # transform them from T1 to MNI space 
        if not os.path.isfile(sin_img_mni_file):
            reference = layout.get(subject=sub, scope='fMRIPrep', datatype='func', space='MNI152NLin2009cAsym', suffix='mask', extension='.nii.gz', return_type='file')[0]
            transform = layout.get(subject=sub, scope='fMRIPrep', datatype='anat', extension='.h5', return_type='file')[3] 
            os.system(f'ANTSENV antsApplyTransforms -i {sin_img_t1_file} -o {sin_img_mni_file} -r {reference} -t {transform}')
            os.system(f'ANTSENV antsApplyTransforms -i {cos_img_t1_file} -o {cos_img_mni_file} -r {reference} -t {transform}')
            
        sin_runs[:,i_run] = nilearn.masking.apply_mask(sin_img_mni_file, mask_file_sig_cluster)
        cos_runs[:,i_run] = nilearn.masking.apply_mask(cos_img_mni_file, mask_file_sig_cluster)
        
    
    # cross-validation: compare orientation between estimation runs and test run
    for run in range(task_info.n_runs): 
        
        # define estimation and test runs
        run_test        = copy.deepcopy(run)
        runs_estimation = np.setdiff1d(np.arange(task_info.n_runs), run)
        
        # orientation in estimation runs
        estimation_sin_beta = sin_runs[:,runs_estimation].mean(axis=1)
        estimation_cos_beta = cos_runs[:,runs_estimation].mean(axis=1)
        estimation_orient = estimate_orientation(estimation_sin_beta, estimation_cos_beta, sym=6)[1]
        
        # orientation in test run
        test_orient = estimate_orientation(sin_runs[:,run_test], cos_runs[:,run_test], sym=6)[1]
        
        # difference between estimation and test orientations in 60°-space        
        orient_diff = np.rad2deg(estimation_orient) - np.rad2deg(test_orient)
        # ori_diff = abs((ori_diff + 180) % 360 - 180) # difference in 360°-space
        orient_diff = abs((orient_diff + 30) % 60 - 30) # difference in 60°-space

        # save for cv fold
        voxels_orient_diff[:,run] = copy.deepcopy(orient_diff)         

    
    # average across cv folds
    voxels_orient_diff_mean = voxels_orient_diff.mean(axis=1)
    
    subject.append(sub)
    condition.append('mean_orientation_difference')
    effect_size.append(voxels_orient_diff_mean.mean())
    
    subject.append(sub)
    condition.append('percentage_stable_voxels')
    effect_size.append((voxels_orient_diff_mean<=criterion_stable_voxel).mean())


data_orient_temp_stab = pd.DataFrame({'subject': subject,
                                      'condition': condition,
                                      'effect_size': effect_size})


#%% DIFFERENCE IN HEXADIRECTIONAL EFFECT BETWEEN HIGH AND LOW VALUE AREAS OF THE SPACE

hexa_value_model_name = 'hexadirectional_analysis_high_vs_low_value_areas'
hexa_value_fl_model_path = os.path.join(valspa.first_level_path, task, hexa_value_model_name)

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
    
    # conditions/contrasts of interest as saved in first level folder
    contrast_imgs = [
        'sym-6_dataset-test_glm-parametric_contrast-parametric_regressor_6_fold_high_value_vs_low_value_effect_size',
        'sym-6_dataset-test_glm-parametric_contrast-parametric_regressor_6_fold_high_value_effect_size',
        'sym-6_dataset-test_glm-parametric_contrast-parametric_regressor_6_fold_low_value_effect_size'
        ]

    # names of conditions for df
    condition_names = [
        '6-fold_glm_parametric_contrast_parametric_high_vs_low_value',
        '6-fold_glm_parametric_contrast_parametric_high_value',
        '6-fold_glm_parametric_contrast_parametric_low_value'
        ]
     
    # extract roi values
    for i_con, con in enumerate(contrast_imgs):
        contrast_file = os.path.join(hexa_value_fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_transform}_{con}.nii.gz') 
        roi_mean = nilearn.masking.apply_mask(contrast_file, mask_file_sig_cluster).mean()
        
        subject.append(sub)
        condition.append(condition_names[i_con])
        effect_size.append(roi_mean)
        

data_hexa_value_split = pd.DataFrame({'subject': subject,
                                      'condition': condition,
                                      'effect_size': effect_size})


#%% EXTRACT DATA FOR HEXADIRECTIONAL ANALYSIS IN VMPFC ROIs

vmpfc_rois = {'vmPFC_value_difference': 'value_difference_vmpfc_peak_sphere_radius_7',
              'vmPFC_Constantinescu': 'vmPFC_Constantinescu_peak_sphere_radius_7'}

# initialize lists for columns in df across subjects
subject     = []
condition   = []
effect_size = []


for sub in valspa.subs:
    
    for roi in vmpfc_rois:
        
        # model and masks based on specific vmPFC ROI
        vmpfc_model_name = f'hexadirectional_analysis_orientation_{roi}'
        vmpfc_fl_model_path = os.path.join(valspa.first_level_path, task, vmpfc_model_name)   
        mask_vmpfc = os.path.join(valspa.derivatives_path, 'masks', 'mni', f'{vmpfc_rois[roi]}.nii.gz')     
     
        # conditions/contrasts of interest as saved in first level folder
        contrast_imgs = ['sym-{sym}_dataset-test_glm-parametric_contrast-parametric_regressor_{sym}_fold_effect_size' for sym in symmetries] 
                
        # names of conditions for df
        condition_names = ['{sym}-fold_glm_parametric_contrast_parametric' for sym in symmetries]
            
        # extract roi values
        for i_con, con in enumerate(contrast_imgs):
            contrast_file = os.path.join(vmpfc_fl_model_path, f'sub-{sub}', f'sub-{sub}_task-{task}_run-all_space-{space_transform}_{con}.nii.gz') 
            roi_mean = nilearn.masking.apply_mask(contrast_file, mask_vmpfc).mean()
            
            subject.append(sub)
            condition.append(f'roi-{roi}_{condition_names[i_con]}')
            effect_size.append(roi_mean)
          

# create dataframe
data_vmpfc = pd.DataFrame({'subject': subject,
                           'condition': condition,
                           'effect_size': effect_size})


#%% SAVE DATA

hexa_data = {'data_significant_cluster': data_significant_cluster,
             'data_fl_roi': data_fl_roi,
             'data_orient_spatial_stab': data_orient_spatial_stab,
             'data_orient_temp_stab': data_orient_temp_stab,
             'data_hexa_value_split': data_hexa_value_split,
             'data_vmpfc': data_vmpfc}

with open(os.path.join(valspa.derivatives_path, 'hexadirectional_analysis', 'group', 'hexadirectional_data.pkl'), 'wb') as f:
    pickle.dump(grid_data, f)


