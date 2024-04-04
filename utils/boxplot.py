# -*- coding: utf-8 -*-
"""
@author: Suraj
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap

cIdx_df = pd.read_csv('c-index-result.csv')
cIdx_df['Data'] = ['\n'.join(wrap(x, 12)) for x in cIdx_df['Data']]

plt.figure(figsize=(15, 10))
box_plot = sns.boxplot(x="Data", y="Cindex", data=cIdx_df, palette="Set2")
box_plot.set_xticklabels(box_plot.get_xticklabels())
box_plot.figure.savefig('Results/boxplot.pdf')
