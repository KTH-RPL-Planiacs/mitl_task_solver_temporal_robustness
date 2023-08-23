
import numpy as np
import pulp as pulp
from itertools import product, combinations
from MTL import MTLFormula
from compute_left_right_time_robustness import left_right_time_robustness, left_right_time_robustness_demands
import time


def timerobust_lp(hist_mdp, H, phi, untimed_S, timeout_at=30000,MAX_RIGHT_TIME_ROBUSTNESS = 10):
    #constrained mdp with a singular, cumulative constraint 

    start_time = time.time()
    initial_state = hist_mdp.get_initial_state()
    states = hist_mdp.get_states()
    states_by_transitions=[[s for s in states if len(s)==t+1] for t in range(H+1)]



    
    reward_dict={}
    for s in states:
        if len(s)==H-1 and s[-1][-1] == H-1:
            reward_dict[s]=test_time_robustness_history(s,phi,untimed_S,H, MAX_RIGHT_TIME_ROBUSTNESS)[1]
        else:
            reward_dict[s]=0
    reward_time= time.time()
    print('reward calculated in', reward_time-start_time)

    model = pulp.LpProblem("TR_LP", pulp.LpMaximize)

    x_tsa = pulp.LpVariable.dicts("x_tsa",
                                     (( t, calc_trace(s,H), a)
                                     for t in range(H)
                                     for s in states_by_transitions[t]
                                     for a in hist_mdp.get_allowable_actions(s)
                                     ),
                                     lowBound=0, upBound=1,
                                     cat='Continuous')
                                     
                                     
    # Objective Function
    model += (
        pulp.lpSum([reward_dict[s]*x_tsa[t,calc_trace(s,H),a] 
        for t in range(H)
        for s in states_by_transitions[t]
        for a in hist_mdp.get_allowable_actions(s)])
    )



    # # Constraints
    

    # # Valid transisitions


    # Valid transisitions
    for t in range(H-1):
        # print(t)
        for s_next in states_by_transitions[t+1]:
            model += ((
                    pulp.lpSum([x_tsa[t+1,calc_trace(s_next,H),a_next] for a_next in hist_mdp.get_allowable_actions(s_next)])
                    - pulp.lpSum([hist_mdp.get_transition_prob(s,a,s_next)*x_tsa[t,calc_trace(s,H),a] 
                    for s in states_by_transitions[t] for a in hist_mdp.get_allowable_actions(s)])
                    ) == 0)



    # # valid initial states
    # print(hist_mdp.get_allowable_actions(initial_state))
    model += pulp.lpSum([x_tsa[0,calc_trace(initial_state,H),a] for a in hist_mdp.get_allowable_actions(initial_state)])== 1
          


    model_time= time.time()
    print('model generated in', model_time-reward_time)
    
    # Solve


    model.solve(pulp.GUROBI_CMD(msg=True))
    solve_time = time.time()
    print('model solved in', solve_time-model_time)
    model.to_json("nonworking_model.json")


    status = pulp.LpStatus[model.status]
    if status == 'Optimal' :
        print('complete')
        status = 1
        policy_dict = {}
        for t in range(H):
            for s in states_by_transitions[t]:
                for a in hist_mdp.get_allowable_actions(s):
                    if a != 't':
                        unnormal_prob = x_tsa[t,calc_trace(s,H),a].varValue
                        if unnormal_prob is not None and unnormal_prob > 1e-5:
                            normalization = 0
                            for a_norm in hist_mdp.get_allowable_actions(s):
                                if abs(x_tsa[t,calc_trace(s,H),a_norm].varValue) > 1e-5:
                                    normalization = normalization + x_tsa[t,calc_trace(s,H),a_norm].varValue
                                    prob = unnormal_prob/normalization
                            if s not in policy_dict:
                                policy_dict[s]= [(a,prob)]
                            else:
                                policy_dict[s].append((a,prob))
        # print(policy_dict)
        for key,value in policy_dict.items():
            print(key)
            print(value)
        total_reward = pulp.value(model.objective)
        print(total_reward) 
        print('total time is', solve_time-start_time)
        return total_reward, policy_dict
    else:
        print('optimisation failed with status: ', model.status)
        return -MAX_RIGHT_TIME_ROBUSTNESS, policy_dict
        





    
def calc_trace(s,H, list=False):
    trace = ['']*H
    for (state, time) in s:
        trace[time]=state  
    if list:
        return trace
    else:
        return tuple(trace)

def test_time_robustness_history(s,phi,S,H, MAX_RIGHT_TIME_ROBUSTNESS):
    trace = calc_trace(s,H, list=True)
    # print(trace)
    # print(S)
    muplus, muminus = left_right_time_robustness(trace,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)
    return muplus, muminus 