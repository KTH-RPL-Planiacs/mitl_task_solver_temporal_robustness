
import numpy as np
import pulp as pulp
from itertools import product, combinations
from MTL import MTLFormula
from WTS import WTS
from MTLTaskAllocSolver import generate_plan_mtl_wts
from compute_left_right_time_robustness import left_right_time_robustness, left_right_time_robustness_demands
import time


def timerobust_lp(hist_mdp, H, Demands, untimed_S, H_receeding, wts_worst, timeout_at=30000,MAX_RIGHT_TIME_ROBUSTNESS = 10,  ):

    #constrained mdp with a singular, cumulative constraint 

    encoding_start_time = time.time()
    initial_state = hist_mdp.get_initial_state()
    states = hist_mdp.get_states()
    states_by_transitions=[[s for s in states if len(s)==t+1] for t in range(H_receeding+1)]

    if H_receeding > H : 
        raise ValueError("receeding horizon cannot be longer than the time horizon")


    
    reward_dict={}
    for s in states:
        if len(s)==H_receeding and s[-1][-1] == H_receeding-1:
            
            reward_dict[s]=test_time_robustness_history(s, Demands, untimed_S, H, H_receeding, wts_worst,  MAX_RIGHT_TIME_ROBUSTNESS )[1]

        else:
            reward_dict[s]=0
    reward_time= time.time()
    # print('reward calculated in', reward_time-start_time)

    model = pulp.LpProblem("TR_LP", pulp.LpMaximize)

    x_tsa = pulp.LpVariable.dicts("x_tsa",
                                     (( t, calc_trace(s,H_receeding), a)
                                     for t in range(H_receeding)
                                     for s in states_by_transitions[t]
                                     for a in hist_mdp.get_allowable_actions(s)
                                     ),
                                     lowBound=0, upBound=1,
                                     cat='Continuous')
                     
                                     
    # Objective Function
    model += (
        pulp.lpSum([reward_dict[s]*x_tsa[t,calc_trace(s,H_receeding),a] 
        for t in range(H_receeding)
        for s in states_by_transitions[t]
        for a in hist_mdp.get_allowable_actions(s)])
    )



    # # Constraints
    constraint_counter = 0 
    

    # # Valid transisitions


    # Valid transisitions
    for t in range(H_receeding-1):
        # print(t)
        for s_next in states_by_transitions[t+1]:
            constraint_counter = constraint_counter  +1 
            model += ((
                    pulp.lpSum([x_tsa[t+1,calc_trace(s_next,H_receeding),a_next] for a_next in hist_mdp.get_allowable_actions(s_next)])
                    - pulp.lpSum([hist_mdp.get_transition_prob(s,a,s_next)*x_tsa[t,calc_trace(s,H_receeding),a] 
                    for s in states_by_transitions[t] for a in hist_mdp.get_allowable_actions(s)])
                    ) == 0)



    # # valid initial states
    # print(hist_mdp.get_allowable_actions(initial_state))
    model += pulp.lpSum([x_tsa[0,calc_trace(initial_state,H_receeding),a] for a in hist_mdp.get_allowable_actions(initial_state)])== 1
    constraint_counter = constraint_counter  +1       


    model_time= time.time()
    
    # Solve


    model.solve(pulp.GUROBI_CMD(msg=True))
    solve_time = time.time()
    # model.to_json("nonworking_model.json")


    status = pulp.LpStatus[model.status]
    if status == 'Optimal' :
        status = 1
        policy_dict = {}
        variable_counter = 0
        for t in range(H_receeding):
            for s in states_by_transitions[t]:
                for a in hist_mdp.get_allowable_actions(s):
                    variable_counter = variable_counter+1
                    if a != 't':
                        unnormal_prob = x_tsa[t,calc_trace(s,H_receeding),a].varValue
                        if unnormal_prob is not None and unnormal_prob > 1e-5:
                            normalization = 0
                            for a_norm in hist_mdp.get_allowable_actions(s):
                                if abs(x_tsa[t,calc_trace(s,H_receeding),a_norm].varValue) > 1e-5:
                                    normalization = normalization + x_tsa[t,calc_trace(s,H_receeding),a_norm].varValue
                                    prob = unnormal_prob/normalization
                            if s not in policy_dict:
                                policy_dict[s]= [(a,prob)]
                            else:
                                policy_dict[s].append((a,prob))
        # print(policy_dict)
        # for key,value in policy_dict.items():
        #     print(key)
        #     print(value)
        print('number of variables', variable_counter)
        print('number of constraints', constraint_counter)
        total_reward = pulp.value(model.objective)
        print('temporal robustness is',total_reward) 
        encoding_time = model_time-encoding_start_time
        print('encoding time is', encoding_time)
        solving_time = solve_time-model_time
        print('solve time is', solving_time)
        data = {'robustness':total_reward, 'policy':policy_dict, 'encoding time':encoding_time, 'solving time':solving_time, 'constraints':constraint_counter, 'variables':variable_counter}
        return data
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

def test_time_robustness_history(s,Demands,S,H,H_receeding , wts_worst, MAX_RIGHT_TIME_ROBUSTNESS):
    if H_receeding == H :
        trace = calc_trace(s,H, list=True)
        # print(trace)
        # print(S)
        right_robustness = 0
        left_robustness = 0
        for (phi, priority) in Demands:
            muplus, muminus = left_right_time_robustness(trace,phi,S,MAX_RIGHT_TIME_ROBUSTNESS)
            right_robustness = right_robustness + muplus*priority
            left_robustness = left_robustness + muminus*priority
        return right_robustness, left_robustness 
    else:
       C_lookahead =  wts_worst.C
       trace_transitions = {}
       for i in range(0,len(s)-1):
           trace_transitions[(s[i][0], s[i][1], s[i+1][0])] = s[i+1][1]-s[i][1]
       for (s1, t, s2), travel_time  in C_lookahead.items():
           if t < H_receeding:
               if trace_transitions.get((s1, t, s2))!=None:
                #    print(s1, t, s2)
                   C_lookahead[(s1, t, s2)] = trace_transitions[(s1, t, s2)]
               else:
                   C_lookahead[(s1, t, s2)] = H+1
       wts_lookahead = WTS(wts_worst.S, wts_worst.s_0, wts_worst.T,wts_worst.AP,wts_worst.L,C_lookahead)
       trajectory, left_robustness = generate_plan_mtl_wts(Demands,wts_lookahead, H, return_optimum=True, suppress_output=True)
    #    print(left_robustness)
       return -1, left_robustness 