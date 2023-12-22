#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# =============================================================================
# VALUE DIFFERENCE ANALYSIS: PAPER FIGURES AND STATISTICS
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

# import value data
with open(os.path.join(valspa.derivatives_path, 'value_effect', 'group', 'value_data.pkl'), 'rb') as f:
    value = pickle.load(f)

# import behavioral data
with open(os.path.join(valspa.behav_results_path, 'DEC', 'behavioral_data.pkl'), 'rb') as f:
    behav = pickle.load(f)
    
# plotting variables
task = 'DEC'
plot_path  = os.path.join(valspa.derivatives_path, 'value_effect', 'group', 'paper_figures')
font_size_axis_text = 8
font_size_axis_title = 9
# for raincloud plots: how much to shift the violin, points and lines
shift = 0.1
# figure size:
fs_stand = [6.4, 4.8]
# 180mm/3figures = 60 per fig, 60mm=2.36inch
fs_rs = (2.36, 1.77)  
fs_lw = (1.5,1.77)


#%% MAIN FIGURE VALUE DIFFERENCE EFFECT (PAPER: FIGURE 4)
# plot individual panel figures and use patchworklib to combine all panels into one figure 
# brain figures don't work with patchworklib, need to be inserted manually

# FIGURE 1
fig1 = (ggplot(value['data_timecourse_vmpfc'])           
        + aes(x='time', y='beta', color='condition', fill='condition')
        + scale_color_manual({'chosen': 'red', 'unchosen': 'blue'}, labels = ['chosen value', 'unchosen value'])
        + geom_smooth(stat='summary', fun_data ='mean_se')
        + labs(x='Time in seconds locked to choice onset', y='Effect size (a.u.)')
        + theme_classic()
        + theme(axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))      
        + theme(legend_title=element_blank(), legend_direction='vertical', legend_text=element_text(colour='black', size=font_size_axis_text))       
        + guides(fill=False)
        )
fig1 
ggsave(plot = fig1, filename = 'fig1.pdf', path = plot_path)
g1 = pw.load_ggplot(fig1, figsize=fs_rs)
g1.set_index('a', size=font_size_axis_title)


# FIGURE 2
value['data_value_rois']['score_mean'] = np.array(behav['data_overall'].score_mean)

corr = valspa.correlation_permutation_test(value['data_value_rois'].value_difference_vmpfc_effect, value['data_value_rois'].score_mean)
corr_txt = f'r = {corr.statistic.round(2)}\np = {corr.pvalue.round(2)}'

fig2 = (ggplot(value['data_value_rois'])
         + aes(x='value_difference_vmpfc_effect', y='score_mean')
         + geom_point(stroke=0, size=1.5)
         + geom_smooth(method='lm', color=valspa.color_petrol, fill=valspa.color_petrol)
         + scale_y_continuous(labels=lambda l: [f'{int(v * 100)}' for v in l], expand=[0,0])
         + coord_cartesian(ylim=(0.7,1))
         + labs(x='Effect size vmPFC cluster (a.u.)', y='Percentage correct choices')
         + theme_classic()
         + theme(axis_text_x = element_text(colour='black'), axis_text_y = element_text(colour='black'), axis_text=element_text(colour='black', size=font_size_axis_text), axis_title=element_text(colour='black', size=font_size_axis_title))
         + annotate(geom='text', x=0.025, y=0.77, label=corr_txt, size=font_size_axis_text)           
        )
fig2
ggsave(plot = fig2, filename = 'fig2.pdf', path = plot_path)
g2 = pw.load_ggplot(fig2, figsize=fs_rs)
g2.set_index('b', size=font_size_axis_title)


# arrange all panels in one figure using patchworklib 
# (insert brain figures manually)
pw.param["margin"]=0.3
g_all = (g1|g2)
g_all.savefig(os.path.join(plot_path,'value_Fig4_composed.pdf'))




