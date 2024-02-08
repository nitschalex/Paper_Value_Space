#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# CHOICE DECODING ANALYSIS: PAPER FIGURES AND STATISTICS
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

# import decoding data
with open(os.path.join(valspa.derivatives_path, 'choice_decoding', 'group', 'decoding_data.pkl'), 'rb') as f:
    decod = pickle.load(f)
    
# import behavioral data
with open(os.path.join(valspa.behav_results_path, 'DEC', 'behavioral_data.pkl'), 'rb') as f:
    behav = pickle.load(f)
    
# plotting variables
task = 'DEC'
plot_path  = os.path.join(valspa.derivatives_path, 'choice_decoding', 'group', 'paper_figures')
font_size_axis_text = 8
font_size_axis_title = 9
# for raincloud plots: how much to shift the violin, points and lines
shift = 0.1
# figure size:
fs_stand = [6.4, 4.8]
# 180mm/3figures = 60 per fig, 60mm=2.36inch
fs_rs = (2.36, 1.77)  
fs_lw = (1.5,1.77)

colors_value = {'low-value':'#9E7D0A', 'high-value':'#256D69'}


# for raincloud plots: function to allow placement of objects left-right (for
# shwoing difference between two conditions)
def alt_sign(x):
    "Alternate +1/-1 if x is even/odd"
    return (-1) ** x


#%% MAIN FIGURE CHOICE DECODING (PAPER: SUPPLEMENTARY FIGURE 8)
# plot individual panel figures and use patchworklib to combine all panels into one figure 
# plot ROI image and logic panel separately 

plot_path_supp  = os.path.join(valspa.derivatives_path, 'choice_decoding', 'group', 'paper_figures', 'supplementary_figure_8')


# FIGURE 1: z-scores for probability difference
decod['data_choice_decoding_all_trials_z'] = decod['data_choice_decoding_all_trials'][decod['data_choice_decoding_all_trials'].condition=='probability_high_vs_low_value_stimulus_zscore']
valspa.one_sample_permutation_t_test(decod['data_choice_decoding_all_trials_z'].effect_size)

fig_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig1 = (ggplot(decod['data_choice_decoding_all_trials_z'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig_m1, style='right', fill=colors_value['high-value'], color=None)
         + geom_jitter(fig_m2, width=shift, height=0, fill=colors_value['high-value'], stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=colors_value['high-value'], color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + scale_x_discrete(limits = ['probability_high_vs_low_value_stimulus_zscore'], labels=['filler'])
         + geom_hline(yintercept = 0)
         + labs(x='Choice decoding', y='z-score probability difference')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1, y=decod['data_choice_decoding_all_trials_z'].effect_size.max()+0.1, label='***', size=font_size_axis_text)         
        )
fig1
ggsave(plot = fig1, filename = 'fig1.pdf', path = plot_path_supp)
g1 = pw.load_ggplot(fig1, figsize=(1.5,1.77))
g1.set_index('c', size=font_size_axis_title)


# FIGURE 2: visualization of probabilities
decod['data_choice_decoding_all_trials_prob'] = decod['data_choice_decoding_all_trials'][(decod['data_choice_decoding_all_trials'].condition=='probability_high_value_stimulus') | (decod['data_choice_decoding_all_trials'].condition=='probability_low_value_stimulus')]

fig2_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
fig2_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_fig2_cond = {'probability_high_value_stimulus': colors_value['high-value'], 'probability_low_value_stimulus': colors_value['low-value']}

fig2 = (ggplot(decod['data_choice_decoding_all_trials_prob'])
         + aes(x='condition', y='effect_size', fill='condition')
         + geom_violin(fig2_m1, style='left-right', color=None)
         + geom_point(fig2_m2, stroke=0, size=1.5)
         + geom_line(fig2_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig2_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['probability_high_value_stimulus', 'probability_low_value_stimulus'], labels = ['high-value', 'low-value'])
         + scale_fill_manual(values = colors_fig2_cond, guide=False)
         + labs(x='Stimulus', y='Decoding probability')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig2 # call figure 
ggsave(plot = fig2, filename = 'fig2.pdf', path = plot_path_supp)
g2 = pw.load_ggplot(fig2, figsize=fs_rs)
g2.set_index('d', size=font_size_axis_title)


# FIGURE 3: correlation with performance
# wide format with all variables for correlations
decod['data_choice_decoding_performance'] = pd.pivot(decod['data_choice_decoding_all_trials'], index='subject', columns='condition', values='effect_size')
decod['data_choice_decoding_performance']['subject'] = decod['data_choice_decoding_performance'].index
decod['data_choice_decoding_performance'].index = decod['data_choice_decoding_performance'].index.rename('index')
y_s = copy.deepcopy(behav['data_overall'][['subject','score_mean']])
decod['data_choice_decoding_performance'] = y_s.merge(decod['data_choice_decoding_performance'], on=['subject']) 

corr = valspa.correlation_permutation_test(decod['data_choice_decoding_performance']['probability_high_vs_low_value_stimulus_zscore'], decod['data_choice_decoding_performance']['score_mean'])
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(2)}'

fig3 = (ggplot(decod['data_choice_decoding_performance'])
         + aes(x='probability_high_vs_low_value_stimulus_zscore', y='score_mean')
         + geom_point(stroke=0, size=1.5)
         + geom_smooth(method='lm', color=colors_value['high-value'], fill=colors_value['high-value'])
         + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + coord_cartesian(ylim=(0.7,1))
         + labs(x='z-score probability difference', y='Percentage correct choices')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=3.5, y=0.77, label=corr_txt, size=font_size_axis_text)           
        )
fig3
ggsave(plot = fig3, filename = 'fig3.pdf', path = plot_path_supp)
g3 = pw.load_ggplot(fig3, figsize=fs_rs)
g3.set_index('e', size=font_size_axis_title)


# FIGURE 4: z-scores for congruent probability difference
decod['data_choice_decoding_all_trials_congruent_z'] = decod['data_choice_decoding_all_trials'][decod['data_choice_decoding_all_trials'].condition=='probability_congruent_high_vs_low_value_stimulus_zscore']
valspa.one_sample_permutation_t_test(decod['data_choice_decoding_all_trials_congruent_z'].effect_size)

fig4_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig4_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig4 = (ggplot(decod['data_choice_decoding_all_trials_congruent_z'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig4_m1, style='right', fill=colors_value['high-value'], color=None)
         + geom_jitter(fig4_m2, width=shift, height=0, fill=colors_value['high-value'], stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=colors_value['high-value'], color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + scale_x_discrete(limits = ['probability_congruent_high_vs_low_value_stimulus_zscore'], labels=['filler'])
         + geom_hline(yintercept = 0)
         + labs(x='Choice decoding', y='z-score congruent probability difference')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1, y=decod['data_choice_decoding_all_trials_congruent_z'].effect_size.max()+0.1, label='***', size=font_size_axis_text)         
        )
fig4
ggsave(plot = fig4, filename = 'fig4.pdf', path = plot_path_supp)
g4 = pw.load_ggplot(fig4, figsize=((1.5, 1.77)))
g4.set_index('f', size=font_size_axis_title)


# FIGURE 5: visualization of congruent stimulus probabilities
decod['data_choice_decoding_all_trials_congruent_prob'] = decod['data_choice_decoding_all_trials'][(decod['data_choice_decoding_all_trials'].condition=='probability_congruent_high_value_stimulus') | (decod['data_choice_decoding_all_trials'].condition=='probability_congruent_low_value_stimulus')]

fig5_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
fig5_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_fig5_cond = {'probability_congruent_high_value_stimulus': colors_value['high-value'], 'probability_congruent_low_value_stimulus': colors_value['low-value']}

fig5 = (ggplot(decod['data_choice_decoding_all_trials_congruent_prob'])
         + aes(x='condition', y='effect_size', fill='condition')
         + geom_violin(fig5_m1, style='left-right', color=None)
         + geom_point(fig5_m2, stroke=0, size=1.5)
         + geom_line(fig5_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig5_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['probability_congruent_high_value_stimulus', 'probability_congruent_low_value_stimulus'], labels = ['congr. high-value', 'congr. low-value'])
         + scale_fill_manual(values = colors_fig5_cond, guide=False)
         + labs(x='Stimulus', y='Decoding probability')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig5 
ggsave(plot = fig5, filename = 'fig5.pdf', path = plot_path_supp)
g5 = pw.load_ggplot(fig5, figsize=fs_rs)
g5.set_index('g', size=font_size_axis_title)


# FIGURE 6: correlation of congruent probability difference with performance
corr = valspa.correlation_permutation_test(decod['data_choice_decoding_performance']['probability_congruent_high_vs_low_value_stimulus_zscore'], decod['data_choice_decoding_performance']['score_mean'])
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(2)}'

fig6 = (ggplot(decod['data_choice_decoding_performance'])
         + aes(x='probability_congruent_high_vs_low_value_stimulus_zscore', y='score_mean')
         + geom_point(stroke=0, size=1.5)
         + geom_smooth(method='lm', color=colors_value['high-value'], fill=colors_value['high-value'])
         + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + coord_cartesian(ylim=(0.7,1))
         + labs(x='z-score congruent probability difference', y='Percentage correct choices')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=2.5, y=0.77, label=corr_txt, size=font_size_axis_text)           
        )
fig6
ggsave(plot = fig6, filename = 'fig6.pdf', path = plot_path_supp)
g6 = pw.load_ggplot(fig6, figsize=fs_rs)
g6.set_index('h', size=font_size_axis_title)


# arrange all panels in one figure using patchworklib 
# (insert roi and logic figure manually)
pw.param["margin"]=0.3
g_all = (g1|g2|g3)/(g4|g5|g6)
g_all.savefig(os.path.join(plot_path_supp,'choice_decoding_Supp_Fig8_composed.pdf'))


#%% CHOICE DECODING - CONTROL SWITCH TIME POINTS (PAPER: SUPPLEMENTARY FIGURE 9)
# plot individual panel figures and use patchworklib to combine all panels into one figure 

plot_path_supp  = os.path.join(valspa.derivatives_path, 'choice_decoding', 'group', 'paper_figures', 'supplementary_figure_9')


# FIGURE 1: z-scores for probability difference
decod['data_choice_decoding_switch_trials_z'] = decod['data_choice_decoding_switch_trials'][decod['data_choice_decoding_switch_trials'].condition=='probability_high_vs_low_value_stimulus_zscore']
valspa.one_sample_permutation_t_test(decod['data_choice_decoding_switch_trials_z'].effect_size)

fig_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig1 = (ggplot(decod['data_choice_decoding_switch_trials_z'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig_m1, style='right', fill=colors_value['high-value'], color=None)
         + geom_jitter(fig_m2, width=shift, height=0, fill=colors_value['high-value'], stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=colors_value['high-value'], color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + scale_x_discrete(limits = ['probability_high_vs_low_value_stimulus_zscore'], labels=['filler'])
         + geom_hline(yintercept = 0)
         + labs(x='Choice decoding', y='z-score probability difference')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1, y=decod['data_choice_decoding_switch_trials_z'].effect_size.max()+0.1, label='***', size=font_size_axis_text)         
        )
fig1
ggsave(plot = fig1, filename = 'fig1.pdf', path = plot_path_supp)
g1 = pw.load_ggplot(fig1, figsize=fs_rs)
g1.set_index('a', size=font_size_axis_title)


# FIGURE 2: visualization of probabilities
decod['data_choice_decoding_switch_trials_prob'] = decod['data_choice_decoding_switch_trials'][(decod['data_choice_decoding_switch_trials'].condition=='probability_high_value_stimulus') | (decod['data_choice_decoding_switch_trials'].condition=='probability_low_value_stimulus')]

fig2_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
fig2_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_fig2_cond = {'probability_high_value_stimulus': colors_value['high-value'], 'probability_low_value_stimulus': colors_value['low-value']}

fig2 = (ggplot(decod['data_choice_decoding_switch_trials_prob'])
         + aes(x='condition', y='effect_size', fill='condition')
         + geom_violin(fig2_m1, style='left-right', color=None)
         + geom_point(fig2_m2, stroke=0, size=1.5)
         + geom_line(fig2_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig2_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['probability_high_value_stimulus', 'probability_low_value_stimulus'], labels = ['high-value', 'low-value'])
         + scale_fill_manual(values = colors_fig2_cond, guide=False)
         + labs(x='Stimulus', y='Decoding probability')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig2 # call figure 
ggsave(plot = fig2, filename = 'fig2.pdf', path = plot_path_supp)
g2 = pw.load_ggplot(fig2, figsize=fs_rs)
g2.set_index('b', size=font_size_axis_title)


# FIGURE 3: correlation with performance
# wide format with all variables for correlations
decod['data_choice_decoding_switch_performance'] = pd.pivot(decod['data_choice_decoding_switch_trials'], index='subject', columns='condition', values='effect_size')
decod['data_choice_decoding_switch_performance']['subject'] = decod['data_choice_decoding_switch_performance'].index
decod['data_choice_decoding_switch_performance'].index = decod['data_choice_decoding_switch_performance'].index.rename('index')
y_s = copy.deepcopy(behav['data_sw_pre_post'].loc[behav['data_sw_pre_post'].switch_pre_post=='switch', ['subject','score_mean']])
decod['data_choice_decoding_switch_performance'] = y_s.merge(decod['data_choice_decoding_switch_performance'], on=['subject']) 

corr = valspa.correlation_permutation_test(decod['data_choice_decoding_switch_performance']['probability_high_vs_low_value_stimulus_zscore'], decod['data_choice_decoding_switch_performance']['score_mean'])
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(3)}'

fig3 = (ggplot(decod['data_choice_decoding_switch_performance'])
         + aes(x='probability_high_vs_low_value_stimulus_zscore', y='score_mean')
         + geom_point(stroke=0, size=1.5)
         + geom_smooth(method='lm', color=colors_value['high-value'], fill=colors_value['high-value'])
         + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + labs(x='z-score probability difference', y='Percentage correct choices at switch')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=3.5, y=0.2, label=corr_txt, size=font_size_axis_text)           
        )
fig3
ggsave(plot = fig3, filename = 'fig3.pdf', path = plot_path_supp)
g3 = pw.load_ggplot(fig3, figsize=fs_rs)
g3.set_index('c', size=font_size_axis_title)


# FIGURE 4: z-scores for congruent probability difference
decod['data_choice_decoding_switch_trials_congruent_z'] = decod['data_choice_decoding_switch_trials'][decod['data_choice_decoding_switch_trials'].condition=='probability_congruent_high_vs_low_value_stimulus_zscore']
valspa.one_sample_permutation_t_test(decod['data_choice_decoding_switch_trials_congruent_z'].effect_size)

fig4_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig4_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig4 = (ggplot(decod['data_choice_decoding_switch_trials_congruent_z'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig4_m1, style='right', fill=colors_value['high-value'], color=None)
         + geom_jitter(fig4_m2, width=shift, height=0, fill=colors_value['high-value'], stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=colors_value['high-value'], color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + scale_x_discrete(limits = ['probability_congruent_high_vs_low_value_stimulus_zscore'], labels=['filler'])
         + geom_hline(yintercept = 0)
         + labs(x='Choice decoding', y='z-score congruent probability difference')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1, y=decod['data_choice_decoding_switch_trials_congruent_z'].effect_size.max()+0.1, label='***', size=font_size_axis_text)         
        )
fig4
ggsave(plot = fig4, filename = 'fig4.pdf', path = plot_path_supp)
g4 = pw.load_ggplot(fig4, figsize=fs_rs)
g4.set_index('d', size=font_size_axis_title)


# FIGURE 5: visualization of congruent stimulus probabilities
decod['data_choice_decoding_switch_trials_congruent_prob'] = decod['data_choice_decoding_switch_trials'][(decod['data_choice_decoding_switch_trials'].condition=='probability_congruent_high_value_stimulus') | (decod['data_choice_decoding_switch_trials'].condition=='probability_congruent_low_value_stimulus')]

fig5_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
fig5_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_fig5_cond = {'probability_congruent_high_value_stimulus': colors_value['high-value'], 'probability_congruent_low_value_stimulus': colors_value['low-value']}

fig5 = (ggplot(decod['data_choice_decoding_switch_trials_congruent_prob'])
         + aes(x='condition', y='effect_size', fill='condition')
         + geom_violin(fig5_m1, style='left-right', color=None)
         + geom_point(fig5_m2, stroke=0, size=1.5)
         + geom_line(fig5_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig5_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['probability_congruent_high_value_stimulus', 'probability_congruent_low_value_stimulus'], labels = ['congr. high-value', 'congr. low-value'])
         + scale_fill_manual(values = colors_fig5_cond, guide=False)
         + labs(x='Stimulus', y='Decoding probability')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig5 
ggsave(plot = fig5, filename = 'fig5.pdf', path = plot_path_supp)
g5 = pw.load_ggplot(fig5, figsize=fs_rs)
g5.set_index('e', size=font_size_axis_title)


# FIGURE 6: correlation of congruent probability difference with performance
corr = valspa.correlation_permutation_test(decod['data_choice_decoding_switch_performance']['probability_congruent_high_vs_low_value_stimulus_zscore'], decod['data_choice_decoding_switch_performance']['score_mean'])
corr_txt = f'r = {corr.statistic.round(2)}\np < .001'

fig6 = (ggplot(decod['data_choice_decoding_switch_performance'])
         + aes(x='probability_congruent_high_vs_low_value_stimulus_zscore', y='score_mean')
         + geom_point(stroke=0, size=1.5)
         + geom_smooth(method='lm', color=colors_value['high-value'], fill=colors_value['high-value'])
         + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + labs(x='z-score congruent probability difference', y='Percentage correct choices at switch')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=3.5, y=0.2, label=corr_txt, size=font_size_axis_text)           
        )
fig6
ggsave(plot = fig6, filename = 'fig6.pdf', path = plot_path_supp)
g6 = pw.load_ggplot(fig6, figsize=fs_rs)
g6.set_index('f', size=font_size_axis_title)


# arrange all panels in one figure 
pw.param["margin"]=0.3
g_all = (g1|g2|g3)/(g4|g5|g6)
g_all.savefig(os.path.join(plot_path_supp,'choice_decoding_Supp_Fig9_composed.pdf'))


#%% CHOICE DECODING - CONTROL INCORRECT TRIALS (PAPER: SUPPLEMENTARY FIGURE 10)
# plot individual panel figures and use patchworklib to combine all panels into one figure 

plot_path_supp  = os.path.join(valspa.derivatives_path, 'choice_decoding', 'group', 'paper_figures', 'supplementary_figure_10')


# FIGURE 1: z-scores for probability difference
decod['data_choice_decoding_incorrect_trials_z'] = decod['data_choice_decoding_incorrect_trials'][decod['data_choice_decoding_incorrect_trials'].condition=='probability_high_vs_low_value_stimulus_zscore']
valspa.one_sample_permutation_t_test(decod['data_choice_decoding_incorrect_trials_z'].effect_size)

fig_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig1 = (ggplot(decod['data_choice_decoding_incorrect_trials_z'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig_m1, style='right', fill=colors_value['high-value'], color=None)
         + geom_jitter(fig_m2, width=shift, height=0, fill=colors_value['high-value'], stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=colors_value['high-value'], color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + scale_x_discrete(limits = ['probability_high_vs_low_value_stimulus_zscore'], labels=['filler'])
         + geom_hline(yintercept = 0)
         + labs(x='Choice decoding', y='z-score probability difference')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1, y=decod['data_choice_decoding_incorrect_trials_z'].effect_size.max()+0.1, label='***', size=font_size_axis_text)         
        )
fig1
ggsave(plot = fig1, filename = 'fig1.pdf', path = plot_path_supp)
g1 = pw.load_ggplot(fig1, figsize=fs_rs)
g1.set_index('a', size=font_size_axis_title)


# FIGURE 2: visualization of probabilities
decod['data_choice_decoding_incorrect_trials_prob'] = decod['data_choice_decoding_incorrect_trials'][(decod['data_choice_decoding_incorrect_trials'].condition=='probability_high_value_stimulus') | (decod['data_choice_decoding_incorrect_trials'].condition=='probability_low_value_stimulus')]

fig2_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
fig2_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_fig2_cond = {'probability_high_value_stimulus': colors_value['high-value'], 'probability_low_value_stimulus': colors_value['low-value']}

fig2 = (ggplot(decod['data_choice_decoding_incorrect_trials_prob'])
         + aes(x='condition', y='effect_size', fill='condition')
         + geom_violin(fig2_m1, style='left-right', color=None)
         + geom_point(fig2_m2, stroke=0, size=1.5)
         + geom_line(fig2_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig2_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['probability_high_value_stimulus', 'probability_low_value_stimulus'], labels = ['high-value', 'low-value'])
         + scale_fill_manual(values = colors_fig2_cond, guide=False)
         + labs(x='Stimulus', y='Decoding probability')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig2 # call figure 
ggsave(plot = fig2, filename = 'fig2.pdf', path = plot_path_supp)
g2 = pw.load_ggplot(fig2, figsize=fs_rs)
g2.set_index('b', size=font_size_axis_title)


# FIGURE 3: z-scores for congruent probability difference
decod['data_choice_decoding_incorrect_trials_congruent_z'] = decod['data_choice_decoding_incorrect_trials'][decod['data_choice_decoding_incorrect_trials'].condition=='probability_congruent_high_vs_low_value_stimulus_zscore']
valspa.one_sample_permutation_t_test(decod['data_choice_decoding_incorrect_trials_congruent_z'].effect_size)

fig3_m1 = aes(x=stage('condition', after_scale='x+shift'))    # shift outward
fig3_m2 = aes(x=stage('condition', after_scale='x-2*shift'))  # shift inward

fig3 = (ggplot(decod['data_choice_decoding_incorrect_trials_congruent_z'])
         + aes(x='condition', y='effect_size')
         + geom_violin(fig3_m1, style='right', fill=colors_value['high-value'], color=None)
         + geom_jitter(fig3_m2, width=shift, height=0, fill=colors_value['high-value'], stroke=0, size=1.5)
         + geom_boxplot(outlier_shape='', width=shift, fill=colors_value['high-value'], color='black')
         + stat_summary(fun_data='mean_se', position=position_nudge(0.1), size=0.3, color='black')
         + scale_x_discrete(limits = ['probability_congruent_high_vs_low_value_stimulus_zscore'], labels=['filler'])
         + geom_hline(yintercept = 0)
         + labs(x='Choice decoding', y='z-score congruent probability difference')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=1, y=decod['data_choice_decoding_incorrect_trials_congruent_z'].effect_size.max()+0.1, label='*', size=font_size_axis_text)         
        )
fig3
ggsave(plot = fig3, filename = 'fig3.pdf', path = plot_path_supp)
g3 = pw.load_ggplot(fig3, figsize=fs_rs)
g3.set_index('c', size=font_size_axis_title)


# FIGURE 4: visualization of congruent stimulus probabilities
decod['data_choice_decoding_incorrect_trials_congruent_prob'] = decod['data_choice_decoding_incorrect_trials'][(decod['data_choice_decoding_incorrect_trials'].condition=='probability_congruent_high_value_stimulus') | (decod['data_choice_decoding_incorrect_trials'].condition=='probability_congruent_low_value_stimulus')]

fig4_m1 = aes(x=stage('condition', after_scale='x+shift*alt_sign(x)'))            
fig4_m2 = aes(x=stage('condition', after_scale='x-shift*alt_sign(x)'), group='subject')
colors_fig4_cond = {'probability_congruent_high_value_stimulus': colors_value['high-value'], 'probability_congruent_low_value_stimulus': colors_value['low-value']}

fig4 = (ggplot(decod['data_choice_decoding_incorrect_trials_congruent_prob'])
         + aes(x='condition', y='effect_size', fill='condition')
         + geom_violin(fig4_m1, style='left-right', color=None)
         + geom_point(fig4_m2, stroke=0, size=1.5)
         + geom_line(fig4_m2, color='grey', size=0.1)
         + geom_boxplot(outlier_shape='', width=shift, color='black')
         + stat_summary(fig4_m1, fun_data='mean_se', size=0.3, color='black', fill='black')
         + scale_x_discrete(limits = ['probability_congruent_high_value_stimulus', 'probability_congruent_low_value_stimulus'], labels = ['congr. high-value', 'congr. low-value'])
         + scale_fill_manual(values = colors_fig4_cond, guide=False)
         + labs(x='Stimulus', y='Decoding probability')
         + theme_classic()
         + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
        )
fig4
ggsave(plot = fig4, filename = 'fig4.pdf', path = plot_path_supp)
g4 = pw.load_ggplot(fig4, figsize=fs_rs)
g4.set_index('d', size=font_size_axis_title)


# arrange all panels in one figure 
pw.param["margin"]=0.3
g_all = (g1|g2)/(g3|g4)
g_all.savefig(os.path.join(plot_path_supp,'choice_decoding_Supp_Fig10_composed.pdf'))







