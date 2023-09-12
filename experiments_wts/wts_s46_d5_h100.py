import sys
sys.path.append("..")
from MTL import MTLFormula
from WTS import WTS
from MTLTaskAllocSolver import generate_plan_mtl_wts

import pulp as plp
import matplotlib.pyplot as plt
import numpy as np
import random





if __name__ == '__main__':

    #planning horizon
    PLANNING_HORIZON = 100

    #The WTS of LV24, floor 4
    S = ['s_0_0','s_1_0','s_2_0','s_3_0','s_4_0','s_5_0','s_6_0','s_7_0','s_8_0','s_0_1','s_1_1','s_2_1','s_3_1','s_4_1','s_5_1','s_6_1','s_7_1','s_8_1','s_9_1','s_0_2','s_1_2','s_3_20','s_3_21','s_5_2','s_8_2','s_9_2','s_0_3','s_1_3','s_2_3','s_3_3','s_4_3','s_5_3','s_6_3','s_7_3','s_8_3','s_9_3','s_0_4','s_1_4','s_2_4','s_3_4','s_4_4','s_5_4','s_6_4','s_7_4','s_8_4','s_9_4']
    s_0 = 's_0_3'
    
    T = [('s_0_0','s_0_1'),('s_1_0','s_1_1'),('s_2_0','s_2_1'),('s_3_0','s_3_1'),('s_4_0','s_4_1'),('s_5_0','s_5_1'),('s_6_0','s_6_1'),('s_7_0','s_7_1'),('s_8_0','s_8_1'),('s_0_1','s_1_1'),('s_1_1','s_2_1'),('s_2_1','s_3_1'),('s_3_1','s_4_1'),('s_4_1','s_5_1'),('s_5_1','s_6_1'),('s_6_1','s_7_1'),('s_7_1','s_8_1'),('s_8_1','s_9_1'),('s_1_1','s_1_2'),('s_0_2','s_1_2'),('s_1_2','s_1_3'),('s_3_1','s_3_20'),('s_3_21','s_3_3'),('s_5_1','s_5_2'),('s_5_2','s_5_3'),('s_8_1','s_8_2'),('s_8_2','s_9_2'),('s_8_2','s_8_3'),('s_0_3','s_1_3'),('s_1_3','s_2_3'),('s_2_3','s_3_3'),('s_3_3','s_4_3'),('s_4_3','s_5_3'),('s_5_3','s_6_3'),('s_6_3','s_7_3'),('s_7_3','s_8_3'),('s_0_3','s_1_3'),('s_8_3','s_9_3'),('s_0_3','s_0_4'),('s_1_3','s_1_4'),('s_2_3','s_2_4'),('s_3_3','s_3_4'),('s_4_3','s_4_4'),('s_5_3','s_5_4'),('s_6_3','s_6_4'),('s_7_3','s_7_4'),('s_8_3','s_8_4'),('s_9_3','s_9_4')]
    for state in S:
        T.append((state,state))
    AP = ['E1','E2','K','P','M1','M2','M1','Jana','Alexis']
    
    C = {}
    for t in range(0,PLANNING_HORIZON):
        for s1, s2 in T:
            C[(s1,t,s2)] = 1
            C[(s2,t,s1)] = 1
            C[(s1,t,s1)] = 1
        C[('s_8_1',t,'s_8_2')] = 2
        C[('s_8_2',t,'s_8_3')] = 2
        C[('s_5_1',t,'s_5_2')] = 3
        C[('s_5_2',t,'s_5_3')] = 3
        C[('s_8_2',t,'s_8_1')] = 2
        C[('s_8_3',t,'s_8_2')] = 2
        C[('s_5_2',t,'s_5_1')] = 3
        C[('s_5_3',t,'s_5_2')] = 3
        C[('s_3_21',t,'s_3_3')] = 3
        C[('s_3_3',t,'s_3_21')] = 3
        C[('s_3_20',t,'s_3_1')] = 3
        C[('s_3_1',t,'s_3_20')] = 3


    
    L = {'s_3_20':'E1','s_3_21':'E2','s_9_2':'K','s_5_2':'P','s_0_2':'M1','s_3_4':'M2','s_9_1':'Jana','s_6_0':'Alexis','s_0_3':'Base'}
    wts = WTS(S,s_0,T,AP,L,C)
    
    

    #Eventually_[4,7] x=1,y=8
    e1 = MTLFormula.Eventually(MTLFormula.Predicate('K'),6,12)
    e2 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),15,18)
    a1 = MTLFormula.Always(MTLFormula.Predicate('E2'),45,55)
    e3 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),60,70)
    e4 = MTLFormula.Eventually(MTLFormula.Predicate('Jana'),80,82)
    

    Demands = [(e1,1),(e2,1),(a1,1),(e3,3),(e4,1)]
    

    trajectory = generate_plan_mtl_wts(Demands,wts,PLANNING_HORIZON)
    
    print(trajectory)


