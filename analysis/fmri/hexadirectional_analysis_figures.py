#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# HEXADIRECTIONAL ANALYSIS: PAPER FIGURES AND STATISTICS
#
# AUTHOR: Alexander Nitsch
# CONTACT: nitsch@cbs.mpg.de
# Max Planck Institute for Human Cognitive and Brain Sciences
# DATE: 2023
#
# =============================================================================
"""

#%% IMPORT PACKAGES

from valspa.project_info import *

import seaborn as sn
import patchworklib as pw
from plotnine.options import set_option
set_option('base_family',  'Arial')


#%% SET VARIABLES

# import hexadirectional data
with open(os.path.join(valspa.derivatives_path, 'hexadirectional_analysis', 'group', 'hexadirectional_data.pkl'), 'rb') as f:
    hexa = pickle.load(f)

# import decoding data
with open(os.path.join(valspa.derivatives_path, 'choice_decoding', 'group', 'decoding_data.pkl'), 'rb') as f:
    decod = pickle.load(f)
    
# import behavioral data
with open(os.path.join(valspa.behav_results_path, 'DEC', 'behavioral_data.pkl'), 'rb') as f:
    behav = pickle.load(f)
    
# plotting variables
task = 'DEC'
plot_path  = os.path.join(valspa.derivatives_path, 'hexadirectional_analysis', 'group', 'paper_figures')
font_size_axis_text = 8
font_size_axis_title = 9
# for raincloud plots: how much to shift the violin, points and lines
shift = 0.1
# figure size:
fs_stand = [6.4, 4.8]
# 180mm/3figures = 60 per fig, 60mm=2.36inch
fs_rs = (2.36, 1.77)  
fs_lw = (1.5,1.77)


#%% MAIN FIGURE HEXADIRECTIONAL EFFECT (PAPER: FIGURE 3)
# plot individual panel figures and use patchworklib to combine all panels into one figure 
# some panel figures don't work with patchworklib, they need to be inserted manually


# FIGURE 1: data points of the significant cluster
fig_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig1 = (ggplot(hexa['data_significant_cluster'][hexa['data_significant_cluster'].condition=='6-fold_glm_parametric_contrast_parametric'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig_m1, style='right', fill=valspa.color_petrol, color=None)
         + geom_jitter(fig_m2, width=shift, height=0, stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + geom_hline(yintercept = 0)
         + labs(x='Hexadirectional modulation', y='Effect size (a.u.)')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))        
        )
fig1
ggsave(plot = fig1, filename = 'fig1.pdf', path = plot_path)
g1 = pw.load_ggplot(fig1, figsize=(1.2,1.77))
g1.set_index('a', size=font_size_axis_title)


# FIGURE 2: visualization of effect with directional bins 
hexa['data_six_fold_bins'] = hexa['data_significant_cluster'][hexa['data_significant_cluster'].condition.str.contains('°')] 

# info abput peak bins
sampled_angles_deg  = copy.deepcopy(valspa.dec.angles)
peaks_aligned       = sampled_angles_deg[::6]
peaks_misaligned    = peaks_aligned + 30
peak_bins = sampled_angles_deg[::3]
bin_order = [f'6-fold_glm_peak_bins_contrast_bin_{i}°' for i in peak_bins]

colors_direction_bins = {}
color_aligned = '#E19926'
color_misaligned = '#5A5C79'
for i in peak_bins:
    if i in peaks_aligned:
        colors_direction_bins[f'6-fold_glm_peak_bins_contrast_bin_{i}°'] = color_aligned
    if i in peaks_misaligned:
        colors_direction_bins[f'6-fold_glm_peak_bins_contrast_bin_{i}°'] = color_misaligned

fig2 = (ggplot(hexa['data_six_fold_bins'])
        + aes(x='condition',y='effect_size', fill='condition')
        + geom_boxplot(outlier_shape='', color='black', width=0.5)
        + stat_summary(fun_data='mean_se', position=position_nudge(0.45), size=0.3)
        + labs(x='Direction bins sorted according to grid orientation', y='Effect size (a.u.)')
        + scale_x_discrete(limits=bin_order, labels=[f'{i}°' for i in peak_bins])
        + scale_y_continuous(limits = [-0.25, 0.25]) # adapt so that whiskers are fully visible
        + scale_fill_manual(values = colors_direction_bins, guide=False)
        + geom_hline(yintercept = 0)
        + theme_classic()
        + theme(axis_text_x = element_text(colour='black', angle=45), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))        
        )
fig2 # call figure
ggsave(plot = fig2, filename = 'fig2.pdf', path = plot_path)
g2 = pw.load_ggplot(fig2, figsize=fs_rs)
g2.set_index('b', size=font_size_axis_title)


# FIGURE 3: control symmetries
syms = [4,5,6,7,8]
for sym in syms:
    test_data = hexa['data_fl_roi'][hexa['data_fl_roi'].condition==f'{sym}-fold'].effect_size
    print(f'{sym}-fold')
    print(valspa.one_sample_permutation_t_test(test_data, alternative='greater'))
    print(test_data.mean())
    print(test_data.std())
    
fig3_m1 = aes(x=stage('condition', after_scale='x+shift'))
fig3_m2 = aes(x=stage('condition', after_scale='x-2*shift'))

fig3 = (ggplot(hexa['data_fl_roi'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig3_m1, style='right', fill=valspa.color_petrol, color=None)
         + geom_jitter(fig3_m2, width=shift, height=0, stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
         + geom_hline(yintercept = 0)
         + labs(x='Symmetry (n-fold modulation)', y='Effect size (a.u.)')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x='6-fold', y=hexa['data_fl_roi'].effect_size.max()+0.005, label='*', size=font_size_axis_text)     
        )
fig3
ggsave(plot = fig3, filename = 'fig3.pdf', path = plot_path)
g3 = pw.load_ggplot(fig3, figsize=fs_rs)
g3.set_index('c', size=font_size_axis_title)

# g3 = pw.load_ggplot(fig2, figsize=fs_rs)
# g3.set_index('c', size=font_size_axis_title)


# FIGURE 4: grid orientations across subjects
hexa['data_orient_across_part'] = hexa['data_orient_spatial_stab'][hexa['data_orient_spatial_stab'].condition=='mean_orientation_amplitude']
hexa['data_orient_mean_across_part'] = scipy.stats.circmean(hexa['data_orient_across_part'].effect_size*6)/6

# combined histogram and scatterplot
bin_size = 30
a , b = np.histogram(np.rad2deg(hexa['data_orient_across_part'].effect_size*6), bins=np.arange(0, 360+bin_size, bin_size))
centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])    
fig4 = plt.figure(figsize=(5,5))
ax = fig4.add_subplot(111, projection='polar')
# ticks will be set at these positions but change labels in order to have labels 
# in range [0°,60°]
ax.set_xticks(np.arange(0,2*np.pi,np.pi/6))
ax.set_xticklabels([f'{string}°' for string in np.arange(0,60,5)])
ax.set_yticks(np.arange(1,a.max()+1))
ax.bar(centers, a, width=np.deg2rad(bin_size), color=valspa.color_petrol, edgecolor='black')
# scatterplot of subjects orientations
# radians with range [0°,60°] must be converted to range [0°,360°] for accurate plotting
ax.scatter(np.array(hexa['data_orient_across_part'].effect_size)*6, np.repeat(a.max() + 0.5, len(np.array(hexa['data_orient_across_part'].effect_size))), color = 'grey')
# mean orientation as arrow
ax.arrow(hexa['data_orient_mean_across_part']*6, 0, 0, np.max(a), width = 0.005, head_width=0.15, head_length=0.1*np.max(a), overhang=1, zorder=2, color='black', length_includes_head=True)
# highlight 45° orientation
ax.arrow(np.deg2rad(45)*6, 0, 0, a.max(), width = 0.005, head_width=0.15, head_length=0.1*np.max(a), overhang=1, zorder=2, color='#B42024', length_includes_head=True)
# plt.savefig(os.path.join(plot_path, 'fig4.pdf), bbox_inches='tight')
plt.show()


# arrange all panels in one figure using patchworklib 
# (insert logic, brain and orientation figure manually)
pw.param["margin"]=0.2
g_all = (g1|g2|g3)
g_all.savefig(os.path.join(plot_path,'hexa_Fig3_composed.pdf'))


#%% MAIN FIGURE HEXADIRECTIONAL EFFECT (PAPER: SUPPLEMENTARY FIGURE 4)
# plot individual panel figures and use patchworklib to combine all panels into one figure 
# some panel figures don't work with patchworklib: first create other objects for those
# and later insert them manually


# FIGURE 1: spatial stability
supp_fig1_m1 = aes(x=stage('condition', after_scale='x+shift'))    
supp_fig1_m2 = aes(x=stage('condition', after_scale='x-2*shift')) 

supp_fig1 = (ggplot(hexa['data_orient_spatial_stab'][hexa['data_orient_spatial_stab'].condition=='rayleigh_z'])
             + aes(x='condition', y='effect_size')
             + geom_violin(supp_fig1_m1, style='right', fill=valspa.color_petrol, color=None)
             + geom_jitter(supp_fig1_m2, width=shift, height=0, stroke=0, size=1.5)
             + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
             + labs(x='Spatial stability', y='Rayleigh z')
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))        
            )
supp_fig1
ggsave(plot = supp_fig1, filename = 'supp_fig1.pdf', path = plot_path)
supp_g1 = pw.load_ggplot(supp_fig1, figsize=fs_rs)
supp_g1.set_index('a', size=font_size_axis_title)
# supp_g2 = pw.load_ggplot(supp_fig1, figsize=fs_rs)
# supp_g2.set_index('a', size=font_size_axis_title)
# supp_g3 = pw.load_ggplot(supp_fig1, figsize=fs_rs)
# supp_g3.set_index('a', size=font_size_axis_title)
# supp_g4 = pw.load_ggplot(supp_fig1, figsize=fs_rs)
# supp_g4.set_index('a', size=font_size_axis_title)


# FIGURE 5: correlation for spatial stability
# wide format with all variables for correlations
hexa['data_six_fold_corr_variables'] = pd.concat([hexa['data_significant_cluster'], hexa['data_orient_spatial_stab'], hexa['data_orient_temp_stab']], ignore_index=True)
hexa['data_six_fold_corr_variables'] = pd.pivot(hexa['data_six_fold_corr_variables'], index='subject', columns='condition', values='effect_size')
hexa['data_six_fold_corr_variables']['subject'] = hexa['data_six_fold_corr_variables'].index
hexa['data_six_fold_corr_variables'].index = hexa['data_six_fold_corr_variables'].index.rename('index')
hexa['data_six_fold_corr_variables'] = hexa['data_six_fold_corr_variables'].reset_index()

corr = valspa.correlation_permutation_test(hexa['data_six_fold_corr_variables']['6-fold_glm_parametric_contrast_parametric'], hexa['data_six_fold_corr_variables']['rayleigh_z'])
corr_txt = f'r = {corr.statistic.round(2)}\np < .001'

supp_fig5 = (ggplot(hexa['data_six_fold_corr_variables'])
             + aes(x='6-fold_glm_parametric_contrast_parametric', y='rayleigh_z')
             + geom_point(stroke=0, size=1.5)
             + geom_smooth(method='lm', color=valspa.color_petrol, fill=valspa.color_petrol)
             + labs(x='Effect size of hexadirectional modulation (a.u.)', y='Rayleigh z')
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x=0.1, y=25, label=corr_txt, size=font_size_axis_text)           
            )
supp_fig5
ggsave(plot = supp_fig5, filename = 'supp_fig5.pdf', path = plot_path)
supp_g5 = pw.load_ggplot(supp_fig5, figsize=fs_rs)
supp_g5.set_index('d', size=font_size_axis_title)


# FIGURE 6: temporal stability
valspa.one_sample_permutation_t_test(hexa['data_orient_temp_stab'][hexa['data_orient_temp_stab'].condition=='percentage_stable_voxels'].effect_size - 0.5)

supp_fig6_m1 = aes(x=stage('condition', after_scale='x+shift'))    
supp_fig6_m2 = aes(x=stage('condition', after_scale='x-2*shift')) 

supp_fig6 = (ggplot(hexa['data_orient_temp_stab'][hexa['data_orient_temp_stab'].condition=='percentage_stable_voxels'])
             + aes(x='condition', y='effect_size')
             + geom_violin(supp_fig6_m1, style='right', fill=valspa.color_petrol, color=None)
             + geom_jitter(supp_fig6_m2, width=shift, height=0, stroke=0, size=1.5)
             + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
             + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
             + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l])
             + labs(x='Temporal stability', y='Percentage stable voxels')
             + geom_hline(yintercept = 0.5)
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x=1, y=1.01, label='**', size=font_size_axis_text)         
            )
supp_fig6
ggsave(plot = supp_fig6, filename = 'supp_fig6.pdf', path = plot_path)
supp_g6 = pw.load_ggplot(supp_fig6, figsize=fs_rs)
supp_g6.set_index('e', size=font_size_axis_title)


# FIGURE 7: correlation for temporal stability
corr = valspa.correlation_permutation_test(hexa['data_six_fold_corr_variables']['6-fold_glm_parametric_contrast_parametric'], hexa['data_six_fold_corr_variables']['percentage_stable_voxels'])
corr_txt = f'r = {corr.statistic.round(2)}\np < .001'

supp_fig7 = (ggplot(hexa['data_six_fold_corr_variables'])
             + aes(x='6-fold_glm_parametric_contrast_parametric', y='percentage_stable_voxels')
             + geom_point(stroke=0, size=1.5)
             + geom_smooth(method='lm', color=valspa.color_petrol, fill=valspa.color_petrol)
             + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
             + coord_cartesian(ylim=(0,1))
             + labs(x='Effect size of hexadirectional modulation (a.u.)', y='Percentage stable voxels')
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x=0.1, y=0.25, label=corr_txt, size=font_size_axis_text)           
            )
supp_fig7
ggsave(plot = supp_fig7, filename = 'supp_fig7.pdf', path = plot_path)
supp_g7 = pw.load_ggplot(supp_fig7, figsize=fs_rs)
supp_g7.set_index('f', size=font_size_axis_title)


# FIGURE 8: correlation with performance
y_s = copy.deepcopy(behav['data_overall'][['subject','score_mean']])
hexa['data_six_fold_corr_variables'] = y_s.merge(hexa['data_six_fold_corr_variables'], on=['subject']) 

corr = valspa.correlation_permutation_test(hexa['data_six_fold_corr_variables']['6-fold_glm_parametric_contrast_parametric'], hexa['data_six_fold_corr_variables']['score_mean'])
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(2)}'

supp_fig8 = (ggplot(hexa['data_six_fold_corr_variables'])
             + aes(x='6-fold_glm_parametric_contrast_parametric', y='score_mean')
             + geom_point(stroke=0, size=1.5)
             + geom_smooth(method='lm', color=valspa.color_petrol, fill=valspa.color_petrol)
             + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
             + coord_cartesian(ylim=(0.7,1))
             + labs(x='Effect size of hexadirectional modulation (a.u.)', y='Percentage correct choices')
             + theme_classic()
             + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x=0.1, y=0.75, label=corr_txt, size=font_size_axis_text)           
            )
supp_fig8
ggsave(plot = supp_fig8, filename = 'supp_fig8.pdf', path = plot_path)
supp_g8 = pw.load_ggplot(supp_fig8, figsize=fs_rs)
supp_g8.set_index('g', size=font_size_axis_title)


# FIGURE 9: directional sampling for value split analysis 
supp_fig9 = (ggplot(hexa['data_value_split_direction_sampling_group'])
                + aes(x='angle',y='mean_value_median_split_count_mean')
                + geom_bar(stat='identity', width=8, fill=valspa.color_petrol) 
                + labs(x='Direction', y='Mean frequency')
                + geom_errorbar(aes(ymin = hexa['data_value_split_direction_sampling_group'].mean_value_median_split_count_mean - hexa['data_value_split_direction_sampling_group'].mean_value_median_split_count_sem, ymax = hexa['data_value_split_direction_sampling_group'].mean_value_median_split_count_mean + hexa['data_value_split_direction_sampling_group'].mean_value_median_split_count_sem), width=3)
                + facet_wrap('~mean_value_median_split')
                + theme_classic()
                + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
                )
supp_fig9
ggsave(plot = supp_fig9, filename = 'supp_fig9.pdf', path = plot_path)
supp_g9 = pw.load_ggplot(supp_fig9, figsize=fs_rs)
supp_g9.set_index('h', size=font_size_axis_title)


# FIGURE 10: value split analysis
valspa.one_sample_permutation_t_test(hexa['data_hexa_value_split'][hexa['data_hexa_value_split'].condition=='6-fold_glm_parametric_contrast_parametric_high_vs_low_value'].effect_size)
valspa.one_sample_permutation_t_test(hexa['data_hexa_value_split'][hexa['data_hexa_value_split'].condition=='6-fold_glm_parametric_contrast_parametric_low_value'].effect_size, alternative='greater')
valspa.one_sample_permutation_t_test(hexa['data_hexa_value_split'][hexa['data_hexa_value_split'].condition=='6-fold_glm_parametric_contrast_parametric_high_value'].effect_size, alternative='greater')

# function to allow placement of objects left-right
def alt_sign(x):
    "Alternate +1/-1 if x is even/odd"
    return (-1) ** x

supp_fig10_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
supp_fig10_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')

supp_fig10 = (ggplot(hexa['data_hexa_value_split'][hexa['data_hexa_value_split'].condition!='6-fold_glm_parametric_contrast_parametric_high_vs_low_value'])
             + aes(x='condition', y='effect_size')
             + geom_violin(supp_fig10_m1, style='left-right', fill=valspa.color_petrol, color=None)
             + geom_point(supp_fig10_m2, stroke=0, size=1.5)
             + geom_line(supp_fig10_m2, color='grey', size=0.1)
             + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
             + stat_summary(supp_fig10_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
             + scale_x_discrete(labels = ['high value', 'low value'])
             + geom_hline(yintercept = 0)
             + labs(x='Value condition', y='Effect size of hexadirectional modulation (a.u.)')
             + theme_classic()
             + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
             + annotate(geom='text', x='6-fold_glm_parametric_contrast_parametric_low_value', y=hexa['data_hexa_value_split'][hexa['data_hexa_value_split'].condition!='6-fold_glm_parametric_contrast_parametric_high_vs_low_value'].effect_size.max()+0.01, label='*', size=font_size_axis_text)     
            )
supp_fig10 
ggsave(plot = supp_fig10, filename = 'supp_fig10.pdf', path = plot_path)
supp_g10 = pw.load_ggplot(supp_fig10, figsize=fs_rs)
supp_g10.set_index('i', size=font_size_axis_title)


# FIGURE 11: vmPFC effects
valspa.one_sample_permutation_t_test(hexa['data_vmpfc'][hexa['data_vmpfc'].condition=='roi-vmPFC_value_difference_6-fold_glm_parametric_contrast_parametric'].effect_size, alternative='greater')
valspa.one_sample_permutation_t_test(hexa['data_vmpfc'][hexa['data_vmpfc'].condition=='roi-vmPFC_Constantinescu_6-fold_glm_parametric_contrast_parametric'].effect_size, alternative='greater')

supp_fig11_m1 = aes(x=stage('condition', after_scale='x+shift'))
supp_fig11_m2 = aes(x=stage('condition', after_scale='x-2*shift'))

supp_fig11 = (ggplot(hexa['data_vmpfc'][(hexa['data_vmpfc'].condition=='roi-vmPFC_value_difference_6-fold_glm_parametric_contrast_parametric') |  (hexa['data_vmpfc'].condition=='roi-vmPFC_Constantinescu_6-fold_glm_parametric_contrast_parametric')])
         + aes(x='condition', y='effect_size')
         + geom_violin(supp_fig11_m1, style='right', fill=valspa.color_petrol, color=None)
         + geom_jitter(supp_fig11_m2, width=shift, height=0, stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=valspa.color_petrol, color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['roi-vmPFC_value_difference_6-fold_glm_parametric_contrast_parametric', 'roi-vmPFC_Constantinescu_6-fold_glm_parametric_contrast_parametric'], labels = ['vmPFC (value effect)', 'vmPFC (Constantinescu et al., 2016)'])
         + geom_hline(yintercept = 0)
         + labs(x='vmPFC ROI', y='Effect size of hexadirectional modulation (a.u.)')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         )
supp_fig11 # call figure 
ggsave(plot = supp_fig11, filename = 'supp_fig11.pdf', path = plot_path)
supp_g11 = pw.load_ggplot(supp_fig11, figsize=fs_rs)
supp_g11.set_index('j', size=font_size_axis_title)
# supp_g12 = pw.load_ggplot(supp_fig11, figsize=fs_rs)
# supp_g12.set_index('j', size=font_size_axis_title)
# supp_g13 = pw.load_ggplot(supp_fig11, figsize=fs_rs)
# supp_g13.set_index('j', size=font_size_axis_title)


# arrange all panels in one figure using patchworklib 
# (insert mask and orientation figure manually)
pw.param["margin"]=0.3
g_all = (supp_g1|supp_g2|supp_g3)/(supp_g4|supp_g5|supp_g6)/(supp_g7|supp_g8|supp_g10)/(supp_g11|supp_g12|supp_g13)
g_all.savefig(os.path.join(plot_path,'hexa_Supp_Fig4_composed.pdf'))








