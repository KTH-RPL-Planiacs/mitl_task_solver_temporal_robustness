import sys
sys.path.append("..")
import time
import random
import pickle

from MDP_Object import MDP
from MTLTaskAllocSolver_MDP2 import timerobust_lp
from MDP_examples import construct_timed_MDP, construct_bounding_WTS, construct_history_MDP
from MTL import MTLFormula
from WTS import WTS






if __name__ == '__main__':

    #planning horizon
    PLANNING_HORIZON = 50
    RECEEDING_HORIZON = 5
    D = 2



    untimed_initial_state = 's_0_3'
    untimed_S= ['s_0_0','s_1_0','s_2_0','s_3_0','s_4_0','s_5_0','s_6_0','s_7_0','s_8_0','s_0_1','s_1_1','s_2_1','s_3_1','s_4_1','s_5_1','s_6_1','s_7_1','s_8_1','s_9_1','s_0_2','s_1_2','s_3_20','s_3_21','s_5_2','s_8_2','s_9_2','s_0_3','s_1_3','s_2_3','s_3_3','s_4_3','s_5_3','s_6_3','s_7_3','s_8_3','s_9_3','s_0_4','s_1_4','s_2_4','s_3_4','s_4_4','s_5_4','s_6_4','s_7_4','s_8_4','s_9_4']
    
    additive_time_transition_dict = {}
    for s in untimed_S:
        additive_time_transition_dict[s] = {}

    actions = []
    T = [('s_0_0','s_0_1'),('s_1_0','s_1_1'),('s_2_0','s_2_1'),('s_3_0','s_3_1'),('s_4_0','s_4_1'),('s_5_0','s_5_1'),('s_6_0','s_6_1'),('s_7_0','s_7_1'),('s_8_0','s_8_1'),('s_0_1','s_1_1'),('s_1_1','s_2_1'),('s_2_1','s_3_1'),('s_3_1','s_4_1'),('s_4_1','s_5_1'),('s_5_1','s_6_1'),('s_6_1','s_7_1'),('s_7_1','s_8_1'),('s_8_1','s_9_1'),('s_1_1','s_1_2'),('s_0_2','s_1_2'),('s_1_2','s_1_3'),('s_3_1','s_3_20'),('s_3_21','s_3_3'),('s_5_1','s_5_2'),('s_5_2','s_5_3'),('s_8_1','s_8_2'),('s_8_2','s_9_2'),('s_8_2','s_8_3'),('s_0_3','s_1_3'),('s_1_3','s_2_3'),('s_2_3','s_3_3'),('s_3_3','s_4_3'),('s_4_3','s_5_3'),('s_5_3','s_6_3'),('s_6_3','s_7_3'),('s_7_3','s_8_3'),('s_0_3','s_1_3'),('s_8_3','s_9_3'),('s_0_3','s_0_4'),('s_1_3','s_1_4'),('s_2_3','s_2_4'),('s_3_3','s_3_4'),('s_4_3','s_4_4'),('s_5_3','s_5_4'),('s_6_3','s_6_4'),('s_7_3','s_7_4'),('s_8_3','s_8_4'),('s_9_3','s_9_4')]
    probablistic_transitions = {('s_8_2','s_8_3'):2, ('s_8_1','s_8_2'):2,('s_5_1','s_5_2'):3, ('s_5_2','s_5_3'):3, ('s_8_2','s_8_1'): 2, ('s_8_3','s_8_2'):2, ('s_5_2','s_5_1'): 3, ('s_5_3','s_5_2'): 3, ('s_3_21','s_3_3'):3, ('s_3_3','s_3_21'):3, ('s_3_20','s_3_1'): 3, ('s_3_1','s_3_20'): 3 }
   
    for (s_1, s_2) in T:
        a = s_1 + '_' + s_2
        actions.append(a)
        if (s_1, s_2) in probablistic_transitions:
            additive_time_transition_dict[s_1][a]={(s_2,1 ):.7, (s_2,probablistic_transitions[(s_1, s_2)]):.3}    
            additive_time_transition_dict[s_2][a]={(s_1,1 ):.7, (s_1,probablistic_transitions[(s_1, s_2)]):.3}    
        else: 
            additive_time_transition_dict[s_1][a]={(s_2,1 ):1}    
            additive_time_transition_dict[s_2][a]={(s_1,1 ):1}    

    for s_1 in untimed_S:
        a =  s_1  + '_' + s_1
        actions.append(a)
        additive_time_transition_dict[s_1][a]={(s_1,1 ):1}    

        


    #Eventually_[4,7] x=1,y=8
    e1 = MTLFormula.Eventually(MTLFormula.Predicate('s_9_2'),6,12)
    e2 = MTLFormula.Eventually(MTLFormula.Predicate('s_3_4'),20,26)
    a1 = MTLFormula.Always(MTLFormula.Predicate('s_3_21'),35,45)
 
    Demands = [(e1,1),(e2,1),(a1,1)][:D]



    start_mdp = time.time()

    timed_mdp = construct_timed_MDP(untimed_initial_state, untimed_S, actions, additive_time_transition_dict,  RECEEDING_HORIZON)
    history_mdp = construct_history_MDP(timed_mdp, untimed_S,  RECEEDING_HORIZON)

    wts_worst, wts_best = construct_bounding_WTS(untimed_initial_state, untimed_S, additive_time_transition_dict,  PLANNING_HORIZON)

    end_mdp = time.time()
    data = timerobust_lp(history_mdp, PLANNING_HORIZON, Demands, untimed_S, RECEEDING_HORIZON, wts_worst)
    data['s'] = 46
    data['D']=D
    data['h']=PLANNING_HORIZON
    data['hr']=RECEEDING_HORIZON
    print(data)
    file_name = 'data/s46_d'+str(D)+'_h'+str(PLANNING_HORIZON)+'_rh'+str(RECEEDING_HORIZON)+'_timestamp'+str(round(time.time()))+'.pickle'
    file = open(file_name, "wb")
    pickle.dump(data, file)


