
# coding: utf-8

# In[1]:

import pandas as pd
import MultiContagion as mc
import igraph
import random
import numpy as np
from matplotlib import pylab as plt
import scipy.stats
import copy
from mpl_toolkits.mplot3d import Axes3D
#import powerlaw
get_ipython().magic(u'matplotlib inline')


# Note:  A 2-dimensional array has two corresponding axes: the first running vertically downwards across rows (axis 0), and the second running horizontally across columns (axis 1)

# In[2]:

#year for which to compute the contagion model.
year = 2015


# In[3]:

#import the aggregate adjacency matrix
aggregate_am = np.genfromtxt ('csv_files/AM4_all_nodes_aggregateNorm'+str(year)+'.csv', delimiter=",")
df_names = pd.read_csv('csv_files/all_country_name4.csv', header=None)
names = list(df_names[0])
Aggregate_g = igraph.Graph.Weighted_Adjacency(list(aggregate_am))
Aggregate_g.vs["name"] = copy.deepcopy(names)


# In[4]:

def countries_starting_num(countries_name_starting, g):
    '''Function takes a list of the strings of countries and returns a list of index of those countries in graph g'''
    c_list = []
    for c in countries_name_starting:
        c_list.append(g.vs["name"].index(c))
    return c_list

countries_name_starting = ["United States", "United Kingdom", "Netherlands", "Luxembourg", "China  P.R.: Hong Kong", "Germany", "France", "China  P.R.: Mainland" ]
countries_starting = countries_starting_num(countries_name_starting, Aggregate_g)


# In[6]:

countries_starting


# 201 and 137 are USA and Netherlands respectively

# In[7]:

STR = Aggregate_g.strength(weights=Aggregate_g.es["weight"])


# In[8]:

STR[201], STR[137]


# In[9]:

PR = Aggregate_g.personalized_pagerank(weights=Aggregate_g.es["weight"])


# In[10]:

PR[201], PR[137]


# In[ ]:



