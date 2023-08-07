'''
 Demo code to illustrate how to solve an MDP with LP in Python.

 Resolution of Howard's taxicab MDP by linear programming.

 Code developed by Philippe Preux, Universite de Lille, France & Inria.
 Originally put online on September 23rd, 2020, on https://ph-preux.github.io/software/
  This version is of September 27th, 2020.

 This code has been developed and is provided to the community only to
 serve as a demonstrator. It might not work on your computer, it mght not
 meet your expectations. In any case, this code has not been made in order
 to cause any harm neither to anyone, nor to any computer, nor to anything.
 That being said, you use this code under your own responsability, and risks.

 This code is freely available under MIT licence.
'''

import numpy as np
import pulp as plp
from itertools import product, combinations
from MTL import MTLFormula
from compute_left_right_time_robustness import left_right_time_robustness, left_right_time_robustness_demands
import time

    
#Horizon
H = 5
MAX_RIGHT_TIME_ROBUSTNESS = 10

#States
# S = ['s_0_0','s_1_0','s_0_1','s_1_1']
S = ['s_1','s_2','s_3']

#Actions
# A = {'s_0_0':['s_0_0','s_1_0','s_0_1'], 's_1_0':['s_1_0','s_0_0','s_1_1'], 's_0_1':['s_0_1','s_0_0','s_1_1'], 's_1_1':['s_1_1','s_1_0','s_0_1']}
A = {'s_1':['s_1','s_2','s_3'], 's_2':['s_1','s_2','s_3'], 's_3':['s_1','s_2','s_3']}

#Probabilities
#P[state,action,time]
# P = {'s_0_0':{'s_0_0':{1:1},'s_1_0':{1:.1, 2:.3, 3:.3, 4:.3},'s_0_1':{2:.2, 3:.6, 4:.1, 5:.1}},
 # 's_1_0':{'s_1_0':{1:1},'s_0_0':{1:.1, 2:.3, 3:.3, 4:.3},'s_1_1':{2:.6, 3:.4}}, 
 # 's_0_1':{'s_0_1':{1:1},'s_0_0':{2:.2, 3:.6, 4:.1, 5:.1},'s_1_1':{1:.5, 2:.2, 3:.3}}, 
 # 's_1_1':{'s_1_1':{1:1},'s_1_0':{2:.5, 3:.5},'s_0_1':{1:.5, 2:.2, 3:.3}}
 # }
P = {'s_1':{'s_1':{1:1},'s_2':{1:.6, 2:.4},'s_3':{1:.2, 2:.8}},
 's_2':{'s_2':{1:1},'s_1':{1:.6, 2:.4},'s_3':{1:.5, 2:.5}}, 
 's_3':{'s_3':{1:1},'s_1':{1:.2, 2:.8},'s_2':{1:.5, 2:.5}}
 }


# print([(t,s,a) for t in range(0,H) for s in S for a in A[s]])

#Dictionary of milp variables
dict_vars = {}

#The demands
phi = MTLFormula.Conjunction([MTLFormula.Eventually(MTLFormula.Predicate('s_3'),3,4),MTLFormula.Eventually(MTLFormula.Predicate('s_2'),2,3)])
# e1 = MTLFormula.Eventually(MTLFormula.Predicate('s_3'),3,5)
# Demands = [(e1,2)]

#initializing the time robustness of the demands
# for phi, priority in Demands:
    # mu_plus_phi_t = plp.LpVariable('muplus_'+str(id(phi))+'_t_'+str(0),cat='Integer')
    # dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)] = mu_plus_phi_t  

model = plp.LpProblem("MIP Model", plp.LpMaximize)

# policies_tsa = plp.LpVariable.dicts("policies_tsa", ((t,s,a) for t in range(H) for s in S for a in A[s]), cat='Binary')






#policy tsah
#including the history in the state space
#does not scale : |S|=3, |A|=3, |P[s][a] = 2|, for all states and actions, H=10  => 332220 states
#does not scale : |S|=3, |A|=3, |P[s][a] = 2|, for all states and actions, H=11  => 1183221 states
#does not scale : |S|=3, |A|=3, |P[s][a] = 2|, for all states and actions, H=12  => 4214106 states
#does not scale : |S|=3, |A|=3, |P[s][a] = 2|, for all states and actions, H=13  => 15008763 states (12 sec to calculate state space)
#does not scale : |S|=3, |A|=3, |P[s][a] = 2|, for all states and actions, H=14  => 53454504 states (46 sec to calculate state space)
#does not scale : |S|=3, |A|=3, |P[s][a] = 2|, for all states and actions, H=20  => not computable states
policies_tsash = []
dict_temporal_robustnesses = {}

initial_state = 's_1'
current_state = initial_state
current_time = 0
history = [('s_1',0)]

print('calc state space')
def recursion(c_time,c_state,c_history):
    global policies_tsash_tuple
    for action in A[c_state]:
        policies_tsash.append((c_time,c_state,action,c_history))
        for t in P[c_state][action]:
            n_time = c_time + t
            if n_time<H:
                n_history = c_history + [(action,n_time)]
                recursion(n_time,action,n_history)

recursion(current_time,current_state,history)

# for p in policies_tsash:
    # print(p)
# print(len(policies_tsash))

policies_tsash_tuple = ((c_time,c_state,action,tuple(c_history)) for c_time,c_state,action,c_history in policies_tsash)
pulp_policies_tsah = plp.LpVariable.dicts("policies_tsah", policies_tsash_tuple, lowBound=0, upBound=1, cat='Continuous')

#selecting possible states (t,s,a,h) with a given t,s,a
# selection = [(c_time,c_state,action,c_history) for c_time,c_state,action,c_history in policies_tsash if (c_time==0 and action=='s_1')]
# for sel in selection:
    # print(sel)
    
#selecting possible states (t,s,a,h) with a given t,s,a
# print([pulp_policies_tsah[(c_time,c_state,action,c_history)] for c_time,c_state,action,c_history in pulp_policies_tsah if (c_time==0 and c_state==initial_state)])
    
    
for c_time,c_state,action,c_history in pulp_policies_tsah:
    # print(c_time,c_state,action,c_history)
    for next_action in A[action]:
        s = ''
        for time in P[c_state][action]:
            s += str((c_time+time, action, next_action, c_history+((action,c_time+time),)))
        # print('\t',s)
    # print()

# print([pulp_policies_tsah[(c_time,c_state,action,c_history)] for c_time,c_state,action,c_history in pulp_policies_tsah if (c_time==0 and c_state==initial_state)]) 


def test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS):
    trace = ['']*H
    for state, time in c_history:
        trace[time]=state
    muplus, muminus = left_right_time_robustness(trace,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)
    return muplus, muminus 
    
def test_time_robustness_demands_history(c_history,Demands,S,MAX_RIGHT_TIME_ROBUSTNESS):
    trace = ['']*H
    for state, time in c_history:
        trace[time]=state
    sum_muplus, sum_muminus = left_right_time_robustness_demands(trace,Demands,S,MAX_RIGHT_TIME_ROBUSTNESS)
    return sum_muplus, sum_muminus 



#defines MILP variables for time robustnesses of each of the traces




# #muplus of phi
# model += (plp.lpSum([test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)[0]*pulp_policies_tsah[H-1,c_state,action,c_history] for _,c_state,action,c_history in policies_tsash]))

#muminus of phi

# for c_time,c_state,action,c_history in pulp_policies_tsah:
    # if c_time==H-1:
        # muplus, muminus = test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)
        # print(muplus, muminus)
# exit()

print('calc robustnesses')
dict_temporal_robustnesses = {}
for c_time,c_state,action,c_history in pulp_policies_tsah:
    if c_time==H-1:
        try:
            var = dict_temporal_robustnesses[c_history]
        except KeyError:
            dict_temporal_robustnesses[c_history] = test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)[1] #muminus
            # dict_temporal_robustnesses[c_history] = test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)[0] #muplus

print('objective function')
#this computes all the time robustnesses of all the trajectories beforehand. total time: |S|=3, |A|=3, |P[s][a] = 2|, H=6, 1m17s
model += plp.lpSum([dict_temporal_robustnesses[c_history]*pulp_policies_tsah[(c_time,c_state,action,c_history)] for c_time,c_state,action,c_history in pulp_policies_tsah if (c_time==H-1)])
# model += plp.lpSum([test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)[1]*pulp_policies_tsah[(c_time,c_state,action,c_history)] for c_time,c_state,action,c_history in pulp_policies_tsah if (c_time==H-1)])


# #sum_muplus of Demands
# model += (plp.lpSum([test_time_robustness_demands_history(c_history,Demands,S,MAX_RIGHT_TIME_ROBUSTNESS)[0]*pulp_policies_tsah[H-1,c_state,action,c_history] for _,c_state,action,c_history in policies_tsash]))

# #sum_muminus of Demands
# model += (plp.lpSum([test_time_robustness_demands_history(c_history,Demands,S,MAX_RIGHT_TIME_ROBUSTNESS)[1]*pulp_policies_tsah[H-1,c_state,action,c_history] for _,c_state,action,c_history in policies_tsash]))



def single_non_zero_variable(model,var_ors):
    binary_vars = []
    for var_or in var_ors:
        b_i = plp.LpVariable('b_or_'+str(var_or),cat='Binary')
        binary_vars.append(b_i)
        model += var_or - b_i <= 0
    model += plp.lpSum([b_i for b_i in binary_vars]) == 1

    



# Valid transisitions in the MDP
# for c_time,c_state,action,c_history in pulp_policies_tsah:
    # for next_action in A[action]:
        # try:
            # model += ( pulp_policies_tsah[(c_time,c_state,action,c_history)] - plp.lpSum([P[c_state][action][time] * pulp_policies_tsah[(c_time+time,action,next_action,c_history+((action,c_time+time),))] for time in P[c_state][action]]) ) == 0
            # # for time in P[c_state][action]:
                # # pulp_policies_tsah[(c_time+time,action,next_action,c_history+((action,c_time+time),))] == pulp_policies_tsah[(c_time,c_state,action,c_history)] * P[c_state][action][time]
        # except KeyError:
            # pass

# Valid transisitions in the MDP
dict_binary_vars_actions = {}
for c_time,c_state,action,c_history in pulp_policies_tsah:
    binary_vars_actions = []
    for next_action in A[action]:
        #TODO: ?? should it be 'b_act_'+str((c_time,c_state,action,next_action) OR 'b_act_'+str((c_time,c_state,action,HISTORY,next_action) ??
        try:
            b_i = dict_binary_vars_actions['b_act_'+str((c_time,c_state,action,next_action))]
        except KeyError:
            b_i = plp.LpVariable('b_act_'+str((c_time,c_state,action,next_action)),cat='Binary')
            dict_binary_vars_actions['b_act_'+str((c_time,c_state,action,next_action))] = b_i
        binary_vars_actions.append(b_i)
        for time in P[c_state][action]:
            #if-then-else product construct of pulp_policies_tsah[(c_time+time,action,next_action,c_history+((action,c_time+time),))] == pulp_policies_tsah[(c_time,c_state,action,c_history)] * P[c_state][action][time] * b_i
            try:
                x_l, x_u = 0,1
                model += x_l * b_i <= pulp_policies_tsah[(c_time+time,action,next_action,c_history+((action,c_time+time),))] <= x_u * b_i
                model += pulp_policies_tsah[(c_time,c_state,action,c_history)] * P[c_state][action][time] - (x_u * (1-b_i)) <= pulp_policies_tsah[(c_time+time,action,next_action,c_history+((action,c_time+time),))] <= pulp_policies_tsah[(c_time,c_state,action,c_history)] * P[c_state][action][time] - (x_l * (1-b_i))
            except KeyError:
                #when c_time+time in (c_time+time,action,next_action,c_history+((action,c_time+time),)) is not in the MDP, that is, over the time horizon
                pass
    print(binary_vars_actions, '\t' , c_time,c_state,action,c_history)
    model += plp.lpSum([b_i for b_i in binary_vars_actions]) == 1
    

# valid initial states
model += plp.lpSum([pulp_policies_tsah[(c_time,c_state,action,c_history)] for c_time,c_state,action,c_history in pulp_policies_tsah if (c_time==0 and c_state==initial_state)]) == 1
single_non_zero_variable(model,[pulp_policies_tsah[(c_time,c_state,action,c_history)] for c_time,c_state,action,c_history in pulp_policies_tsah if (c_time==0 and c_state==initial_state)])


model.solve(plp.GUROBI_CMD(msg=True))

for c_time,c_state,action,c_history in pulp_policies_tsah:
    if pulp_policies_tsah[c_time,c_state,action,c_history].varValue:
        if pulp_policies_tsah[c_time,c_state,action,c_history].varValue > 0.0:
            if c_time == H-1:
                print(c_time,c_state,action,c_history, pulp_policies_tsah[c_time,c_state,action,c_history].varValue, '\t timerobustness=',test_time_robustness_history(c_history,phi,S,MAX_RIGHT_TIME_ROBUSTNESS))
            else:
                print(c_time,c_state,action,c_history, pulp_policies_tsah[c_time,c_state,action,c_history].varValue)

exit()




# muplus, muminus = left_right_time_robustness(['s_1','','s_2','s_2','s_2','','','s_3','s_3','s_3','s_3','s_3','s_3','s_3','','','s_2','s_2','','','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','','','s_2','s_2','s_2','s_2','','','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1','s_1'],MTLFormula.Eventually(MTLFormula.Predicate('s_3'),9,12),['s_1','s_2','s_3'],20)
# print(muplus, muminus)

# muplus, muminus = left_right_time_robustness(trace,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)
    
    
    
    
    
    
exit()



# Objective Function
opt_model += plp.lpSum([proba(episode) * plp.lpSum([dict_vars['muplus_'+str(id(phi))+'_t_'+str(0)]*priority for phi, priority in Demands]) for episode in episodes(policy) ])


exit()
 
 
 
 
 

#Timed States
T_S = []
#Probabilities over timed states
T_P = {}
for s in S:
    for t in range(0,H+1):
        T_S.append(s+'_t_'+str(t))

print(T_S)
print(len(T_S))


initial_state = 's_0_0'
S_MDP = {}
S_MDP_actions = {}

t=0
trace = []

def add_s_mdp(state,time,mdp_states,H,P,S_MDP_actions,trace):
    for successor in P[state]:
        for t_plus in P[state][successor]:
            if time<H:
                if successor+'_t_'+str(time+t_plus) not in mdp_states:
                    mdp_states[successor+'_t_'+str(time+t_plus)] = None
                    if time+t_plus>H:
                        S_MDP_actions[successor+'_t_'+str(time+t_plus)] = []
                    else:
                        S_MDP_actions[successor+'_t_'+str(time+t_plus)] = list(P[successor])
                    add_s_mdp(successor,time+t_plus,mdp_states,H,P,S_MDP_actions,trace)
                
                
S_MDP[initial_state+'_t_'+str(t)] = None
S_MDP_actions[initial_state+'_t_'+str(t)] = list(P[initial_state])

add_s_mdp(initial_state,t,S_MDP,H,P,S_MDP_actions,trace)

S_MDP = list(S_MDP)
print(S_MDP, len(S_MDP))

print(S_MDP_actions)

exit()


policies = []

l_states = []
l_actions = []
for state in S_MDP_actions:
    l_states.append(state)
    if S_MDP_actions[state]:
        l_actions.append(S_MDP_actions[state])
    else:
        l_actions.append([None])

print(l_states)
print(l_actions)

# print(*l_actions)

# for el in product(*l_actions):
    # print(el)

# for i,j in product(l_states,l_actions):
    # print(i,j)



# for state in S_MDP_actions:
    # for actions in S_MDP_actions[state]:
    

# for word, letter, hours in product(S_MDP_actions, ["M", "Q", "S", "K", "B"], ["00:00", "01:00", "02:00", "03:00"]):
    # print word, letter, hours

#nothing in list = no product
# for a in product(*l_actions):
    # print(a)
    
product(*l_actions)

print("\n\n\n\n")

act_small   = [['s_0_0', 's_1_0', 's_0_1'], ['s_1_0', 's_0_0', 's_1_1'], [None]]
state_small = ['s_0_0_t_0', 's_0_0_t_1', 's_0_0_t_2']

print(list(product(*act_small)))

print("\n\n\n\n")

for p in product(*act_small):
    # for a in zip(state_small,p):
        # print(a)
    print(list(zip(state_small,p)))

pol = [list(zip(state_small,p)) for p in product(*act_small)]

print(pol)

for p in pol:
    print(p)
    
# print("\n\n\n\n")

i=0
for p in product(*l_actions):
    print(p)
    print(i)
    i += 1

# policies = [list(zip(l_states,p)) for p in product(*l_actions)]

# print(len(policies))

exit()

# definition of the MDP
N = 3 # nb of states
A = 3 # nb of actions
# the transition tensor: P [current state, action, next state] = probability of transiting to next state when emitting action in current state
P = np.array ([[[1/2, 1/4, 1/4],
                [1/16, 3/4, 3/16],
                [1/4, 1/8, 5/8]], 
               [[1/2, 0, 1/2],
                [0,0,0],
                [1/16, 7/8, 1/16]],
               [[1/4, 1/4, 1/2], 
                [1/8, 3/4, 1/8], 
                [3/4, 1/16, 3/16]]])
# the return function: expected return when transiting to next state when emitting action in current state
R = np.array ([10, 4, 8, 8, 2, 4, 4, 6, 4, 14, 0, 18, 0, 0, 0, 8, 16, 8, 10, 2, 8, 6, 4, 2, 4, 0, 8])
R = R.reshape ([N, A, N])

print("R", R)

gamma = .9
# in states 0 and 2, all 3 actions are possible. In state 1, only actions 0 and 2 are possibles:
possible_actions = np.array ([[0, 1, 2], [0, 2], [0, 1, 2]])

# define the LP
v = pulp.LpVariable.dicts ("s", (range (N))) # the variables
prob = pulp.LpProblem ("taxicab", pulp.LpMinimize) # minimize the objective function
prob += sum ([v [i] for i in range (N)]) # defines the objective function
# now, we define the constrain: there is one for each (state, action) pair.
for i in range (N):
    for a in possible_actions [i]:
        prob += v [i] - gamma * sum (P [i, a, j] * v [j] for j in range(N)) >= sum (P [i, a, j] * R [i, a, j] for j in range(N))

# Solve the LP
prob.solve ()
# after resolution, the status of the solution is available and can be printed:
# print("Status:", pulp.LpStatus[prob.status])
# in this case, it should print "Optimal"

# extract the value function
V = np.zeros (N) # value function
for i in range (N):
    V [i] = v [i]. varValue

# extract the optimal policy
pi_star = np.zeros ((N), dtype=np.int64)
vduales = np.zeros ((N, 3))
s = 0
a = 0
for name, c in list(prob.constraints.items()):
    vduales [s, a] = c.pi
    if a < A - 1:
        a = a + 1
    else:
        a = 0
        if s < N - 1:
            s = s + 1
        else:
            s = 0
for s in range(N):
    pi_star [s] = np.argmax (vduales [s, :])

print("pi_star",pi_star)
print("V",V)
for name, c in list(prob.constraints.items()):
    print(name, c)