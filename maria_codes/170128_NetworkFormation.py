
# coding: utf-8

# In[1]:

#Update, add error bars to plots


# In[28]:

import pandas as pd
import MultiContagion as mc
import igraph
import random
import numpy as np
from matplotlib import pylab as plt
import copy
get_ipython().magic(u'matplotlib inline')


# In[29]:

pd.read_stata("CPIS_9countries.dta")


# In[30]:

pd.read_stata("CDIS_9countries.dta")


# * I_A_D_TT - is Assets, Debt Securities
# * I_A_E_TT - is Assets, Equity Securities
# * I_A_T_TT - is Assets, Total Securities
# * I_L_D_TT - is Liabilities, Debt Securities
# * I_L_E_TT - is Liabilities, Equity Securities
# * I_L_T_TT - is Liabilities, Total Securities
# * Asset side is reported by most countries, liability side reporting is voluntary.

# ## Mapping CPIS and obtaining basic measurements

# In[31]:

df_cp = pd.read_stata("CPIS_9countries.dta")


# In[32]:

edges_cp_debt, weight_cp_debt = mc.make_edge_list(df_cp["country"], df_cp["counterpart"], df_cp["I_A_D_T_T_BP6_USD"])


# In[33]:

G_cp_debt = mc.make_graph_from_edge(edges_cp_debt, weight_cp_debt)


# In[34]:

G_cp_debt.strength(weights=G_cp_debt.es["weight"]);


# In[35]:

G_cp_debt.vs["name"];


# In[36]:

country_name = copy.deepcopy(G_cp_debt.vs["name"])


# In[37]:

max_weight = max(G_cp_debt.es["weight"])
E_width_cp = [0.2 + 5*G_cp_debt.es["weight"][i]/max_weight for i in range(len(G_cp_debt.es["weight"]) )]


# In[38]:

#random.seed(105)


# In[39]:

P_cp_debt = igraph.plot(G_cp_debt, vertex_label = G_cp_debt.vs["name"], bbox = (450, 450), margin = (100, 100, 100, 100), edge_width = E_width_cp )


# In[40]:

#P_cp_debt.save("Toy_CPIS_debt.png")


# In[41]:

#random.seed(105)


# In[42]:

edges_cp_equity, weight_cp_equity = mc.make_edge_list(df_cp["country"], df_cp["counterpart"], df_cp["I_A_E_T_T_BP6_USD"])
G_cp_equity = mc.make_graph_from_edge(edges_cp_equity, weight_cp_equity)
max_weight = max(G_cp_equity.es["weight"])
E_width_cp = [0.2 + 5*G_cp_equity.es["weight"][i]/max_weight for i in range(len(G_cp_equity.es["weight"]) )]


# In[43]:

P_cp_equity = igraph.plot(G_cp_equity, vertex_label = G_cp_equity.vs["name"], bbox = (450, 450), margin = (100, 100, 100, 100), edge_width = E_width_cp )


# In[44]:

#P_cp_equity.save("Toy_CPIS_equity.png")


# In[45]:

str_cp_debt = G_cp_debt.strength(weights=G_cp_debt.es["weight"])
in_str_cp_debt = G_cp_debt.strength(weights=G_cp_debt.es["weight"], mode = "IN")
out_str_cp_debt = G_cp_debt.strength(weights=G_cp_debt.es["weight"], mode = "OUT")


# In[46]:

names = list(reversed([x for (y, x) in sorted(zip(in_str_cp_debt, country_name))]))


# In[47]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(in_str_cp_debt, country_name))]))
in_str_cp_debt_ordered = list(reversed(sorted(in_str_cp_debt)))
ax1.bar(np.arange(9), in_str_cp_debt_ordered, 1)
plt.title("In-strength distribution \n of debt assests", fontsize = 20)
plt.xticks(0.5 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
plt.xlabel("Country", fontsize = 12)
plt.ylabel("In-strength", fontsize = 12)
#plt.savefig("instr_cp_debt.png",bbox_inches='tight')
plt.show()


# In[56]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(out_str_cp_debt, country_name))]))
out_str_cp_debt_ordered = list(reversed(sorted(out_str_cp_debt)))
ax1.bar(np.arange(9), out_str_cp_debt_ordered, 1, color = "green")
plt.title("Out-strength distribution \n of debt assets", fontsize = 20)
plt.xticks(0.5 + np.arange(9),names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylim([0, 5e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("Out-strength", fontsize = 12)
#plt.savefig("outstr_cp_debt.png",bbox_inches='tight')
plt.show()


# In[8]:

str_cp_equity = G_cp_debt.strength(weights=G_cp_equity.es["weight"])
in_str_cp_equity = G_cp_debt.strength(weights=G_cp_equity.es["weight"], mode = "IN")
out_str_cp_equity = G_cp_debt.strength(weights=G_cp_equity.es["weight"], mode = "OUT")


# In[57]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(in_str_cp_equity, country_name))]))
in_str_cp_equity_ordered = list(reversed(sorted(in_str_cp_equity)))
ax1.bar(np.arange(9), in_str_cp_equity_ordered, 1)
plt.title("In-strength distribution \n of equity assets", fontsize = 20)
plt.xticks(0.5 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylim([0, 6e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("In-strength", fontsize = 12)
#plt.savefig("instr_cp_equity.png",bbox_inches='tight')
plt.show()


# In[12]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(out_str_cp_equity, country_name))]))
out_str_cp_equity_ordered = list(reversed(sorted(out_str_cp_equity)))
ax1.bar(np.arange(9), out_str_cp_equity_ordered, 1, color = "green")
plt.title("Out-strength distribution \n of equity assets", fontsize = 20)
plt.xticks(0.5 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
plt.ylim([0, 6e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("out-strength", fontsize = 12)
#plt.savefig("outstr_cp_equity.png",bbox_inches='tight')
plt.show()


# ## Contagion on CPIS

# In[26]:

mc.LTM_contagion_time(G_cp_debt, lam = 0.25, chosen_one=2)


# In[48]:

lam_list = [0.25, 0.5, 0.75]
T_iter = 100
n_countries = 9


# In[53]:

def make_several_contagion_list(G, contagion_time = mc.SI_contagion_time, parameters = [0.25, 0.50, 0.75], n_countries = 9, T_iter = 100):
    cont_time_mean_list = []
    cont_time_std_list = [] 
    for i in range(len(parameters)):
        cont_time_mean_list.append([])
        cont_time_std_list.append([])
        lam = parameters[i]
        for node in range(n_countries):
            m, s = contagion_time(G_cp_debt, lam = lam, chosen_one= node, iterations = T_iter)
            cont_time_mean_list[i].append(m)
            cont_time_std_list[i].append(s)
    
    return cont_time_mean_list, cont_time_std_list


# In[54]:

SI_cont_time_mean_list, SI_cont_time_std_list  = make_several_contagion_list(G_cp_debt)


# In[38]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(SI_cont_time_mean_list[0], country_name))]))
SI_cont_time_mean_list_ordered2 = list(reversed([x for (y, x) in sorted(zip(SI_cont_time_mean_list[0], SI_cont_time_mean_list[1]))]))
SI_cont_time_mean_list_ordered3 = list(reversed([x for (y, x) in sorted(zip(SI_cont_time_mean_list[0], SI_cont_time_mean_list[2]))]))
SI_cont_time_mean_list_ordered1 = list(reversed(sorted(SI_cont_time_mean_list[0])))
ax1.plot(SI_cont_time_mean_list_ordered1, "o-", label = "lambda = 0.25")
ax1.plot(SI_cont_time_mean_list_ordered2, "o-", label = "lambda = 0.50")
ax1.plot(SI_cont_time_mean_list_ordered3, "o-", label = "lambda = 0.75")
plt.title("Contagion time in SI model \n of CP debt assets network", fontsize = 20)
plt.xticks(0.1 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
#plt.ylim([0, 6e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("Time steps", fontsize = 12)
plt.legend()
#plt.savefig("SIcont-time_cp_debt.png",bbox_inches='tight')
plt.show()


# In[52]:

LTM_cont_time_std_list 


# In[51]:

LTM_cont_time_mean_list, LTM_cont_time_std_list  = make_several_contagion_list(G_cp_debt, contagion_time=mc.LTM_contagion_time, parameters= [1, 2, 3] )


# In[40]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(LTM_cont_time_mean_list[0], country_name))]))
LTM_cont_time_mean_list_ordered2 = list(reversed([x for (y, x) in sorted(zip(LTM_cont_time_mean_list[0], LTM_cont_time_mean_list[1]))]))
LTM_cont_time_mean_list_ordered3 = list(reversed([x for (y, x) in sorted(zip(LTM_cont_time_mean_list[0], LTM_cont_time_mean_list[2]))]))
LTM_cont_time_mean_list_ordered1 = list(reversed(sorted(LTM_cont_time_mean_list[0])))
ax1.plot(LTM_cont_time_mean_list_ordered1, "o-", label = "phi = 1")
ax1.plot(LTM_cont_time_mean_list_ordered2, "o-", label = "phi = 2")
ax1.plot(LTM_cont_time_mean_list_ordered3, "o-", label = "phi = 3")
plt.title("Contagion time in LTM model \n of CP debt assets network", fontsize = 20)
plt.xticks(0.1 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
#plt.ylim([0, 6e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("Time steps", fontsize = 12)
plt.legend()
#plt.savefig("LTMcont-time_cp_debt.png",bbox_inches='tight')
plt.show()


# ### For Equity data

# In[ ]:

SI_cont_time_mean_list, SI_cont_time_std_list  = make_several_contagion_list(G_cp_equity)


# In[36]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(SI_cont_time_mean_list[0], country_name))]))
SI_cont_time_mean_list_ordered2 = list(reversed([x for (y, x) in sorted(zip(SI_cont_time_mean_list[0], SI_cont_time_mean_list[1]))]))
SI_cont_time_mean_list_ordered3 = list(reversed([x for (y, x) in sorted(zip(SI_cont_time_mean_list[0], SI_cont_time_mean_list[2]))]))
SI_cont_time_mean_list_ordered1 = list(reversed(sorted(SI_cont_time_mean_list[0])))
ax1.plot(SI_cont_time_mean_list_ordered1, "o-", label = "lambda = 0.25")
ax1.plot(SI_cont_time_mean_list_ordered2, "o-", label = "lambda = 0.50")
ax1.plot(SI_cont_time_mean_list_ordered3, "o-", label = "lambda = 0.75")
plt.title("Contagion time in SI model \n of CP equity assets network", fontsize = 20)
plt.xticks(0.1 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
#plt.ylim([0, 6e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("Time steps", fontsize = 12)
plt.legend()
plt.savefig("SIcont-time_cp_equity.png",bbox_inches='tight')
plt.show()


# In[33]:

LTM_cont_time_mean_list, LTM_cont_time_std_list  = make_several_contagion_list(G_cp_equity, contagion_time=mc.LTM_contagion_time, parameters= [1, 2, 3] )


# In[35]:

plt.figure(figsize=(10,10))
fig, ax = plt.subplots()
ax1 = plt.gca()
names = list(reversed([x for (y, x) in sorted(zip(LTM_cont_time_mean_list[0], country_name))]))
LTM_cont_time_mean_list_ordered2 = list(reversed([x for (y, x) in sorted(zip(LTM_cont_time_mean_list[0], LTM_cont_time_mean_list[1]))]))
LTM_cont_time_mean_list_ordered3 = list(reversed([x for (y, x) in sorted(zip(LTM_cont_time_mean_list[0], LTM_cont_time_mean_list[2]))]))
LTM_cont_time_mean_list_ordered1 = list(reversed(sorted(LTM_cont_time_mean_list[0])))
ax1.plot(LTM_cont_time_mean_list_ordered1, "o-", label = "phi = 1")
ax1.plot(LTM_cont_time_mean_list_ordered2, "o-", label = "phi = 2")
ax1.plot(LTM_cont_time_mean_list_ordered3, "o-", label = "phi = 3")
plt.title("Contagion time in LTM model \n of CP equity assets network", fontsize = 20)
plt.xticks(0.1 + np.arange(9), names, rotation='vertical', fontsize = 12)
plt.yticks(fontsize = 12)
#plt.ylim([0, 6e7])
plt.xlabel("Country", fontsize = 12)
plt.ylabel("Time steps", fontsize = 12)
plt.legend()
plt.savefig("LTMcont-time_cp_equity.png",bbox_inches='tight')
plt.show()


# In[ ]:



