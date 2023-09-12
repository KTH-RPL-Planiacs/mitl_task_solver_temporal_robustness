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
    PLANNING_HORIZON = 500

    #The WTS of LV24, floor 4
    S = ['s_0_0','s_1_0','s_2_0','s_3_0','s_4_0','s_5_0','s_6_0','s_7_0','s_8_0','s_0_1','s_1_1','s_2_1','s_3_1','s_4_1','s_5_1','s_6_1','s_7_1','s_8_1','s_9_1','s_0_2','s_1_2','s_3_20','s_3_21','s_5_2','s_8_2','s_9_2','s_0_3','s_1_3','s_2_3','s_3_3','s_4_3','s_5_3','s_6_3','s_7_3','s_8_3','s_9_3','s_0_4','s_1_4','s_2_4','s_3_4','s_4_4','s_5_4','s_6_4','s_7_4','s_8_4','s_9_4','f_1_s_0_0','f_1_s_1_0','f_1_s_2_0','f_1_s_3_0','f_1_s_4_0','f_1_s_5_0','f_1_s_6_0','f_1_s_7_0','f_1_s_8_0','f_1_s_0_1','f_1_s_1_1','f_1_s_2_1','f_1_s_3_1','f_1_s_4_1','f_1_s_5_1','f_1_s_6_1','f_1_s_7_1','f_1_s_8_1','f_1_s_9_1','f_1_s_0_2','f_1_s_1_2','f_1_s_3_20','f_1_s_3_21','f_1_s_5_2','f_1_s_8_2','f_1_s_9_2','f_1_s_0_3','f_1_s_1_3','f_1_s_2_3','f_1_s_3_3','f_1_s_4_3','f_1_s_5_3','f_1_s_6_3','f_1_s_7_3','f_1_s_8_3','f_1_s_9_3','f_1_s_0_4','f_1_s_1_4','f_1_s_2_4','f_1_s_3_4','f_1_s_4_4','f_1_s_5_4','f_1_s_6_4','f_1_s_7_4','f_1_s_8_4','f_1_s_9_4']
    s_0 = 's_0_3'
    
    T = [('s_0_0','s_0_1'),('s_1_0','s_1_1'),('s_2_0','s_2_1'),('s_3_0','s_3_1'),('s_4_0','s_4_1'),('s_5_0','s_5_1'),('s_6_0','s_6_1'),('s_7_0','s_7_1'),('s_8_0','s_8_1'),('s_0_1','s_1_1'),('s_1_1','s_2_1'),('s_2_1','s_3_1'),('s_3_1','s_4_1'),('s_4_1','s_5_1'),('s_5_1','s_6_1'),('s_6_1','s_7_1'),('s_7_1','s_8_1'),('s_8_1','s_9_1'),('s_1_1','s_1_2'),('s_0_2','s_1_2'),('s_1_2','s_1_3'),('s_3_1','s_3_20'),('s_3_21','s_3_3'),('s_5_1','s_5_2'),('s_5_2','s_5_3'),('s_8_1','s_8_2'),('s_8_2','s_9_2'),('s_8_2','s_8_3'),('s_0_3','s_1_3'),('s_1_3','s_2_3'),('s_2_3','s_3_3'),('s_3_3','s_4_3'),('s_4_3','s_5_3'),('s_5_3','s_6_3'),('s_6_3','s_7_3'),('s_7_3','s_8_3'),('s_0_3','s_1_3'),('s_8_3','s_9_3'),('s_0_3','s_0_4'),('s_1_3','s_1_4'),('s_2_3','s_2_4'),('s_3_3','s_3_4'),('s_4_3','s_4_4'),('s_5_3','s_5_4'),('s_6_3','s_6_4'),('s_7_3','s_7_4'),('s_8_3','s_8_4'),('s_9_3','s_9_4'),('f_1_s_0_0','f_1_s_0_1'),('f_1_s_1_0','f_1_s_1_1'),('f_1_s_2_0','f_1_s_2_1'),('f_1_s_3_0','f_1_s_3_1'),('f_1_s_4_0','f_1_s_4_1'),('f_1_s_5_0','f_1_s_5_1'),('f_1_s_6_0','f_1_s_6_1'),('f_1_s_7_0','f_1_s_7_1'),('f_1_s_8_0','f_1_s_8_1'),('f_1_s_0_1','f_1_s_1_1'),('f_1_s_1_1','f_1_s_2_1'),('f_1_s_2_1','f_1_s_3_1'),('f_1_s_3_1','f_1_s_4_1'),('f_1_s_4_1','f_1_s_5_1'),('f_1_s_5_1','f_1_s_6_1'),('f_1_s_6_1','f_1_s_7_1'),('f_1_s_7_1','f_1_s_8_1'),('f_1_s_8_1','f_1_s_9_1'),('f_1_s_1_1','f_1_s_1_2'),('f_1_s_0_2','f_1_s_1_2'),('f_1_s_1_2','f_1_s_1_3'),('f_1_s_3_1','f_1_s_3_20'),('f_1_s_3_21','f_1_s_3_3'),('f_1_s_5_1','f_1_s_5_2'),('f_1_s_5_2','f_1_s_5_3'),('f_1_s_8_1','f_1_s_8_2'),('f_1_s_8_2','f_1_s_9_2'),('f_1_s_8_2','f_1_s_8_3'),('f_1_s_0_3','f_1_s_1_3'),('f_1_s_1_3','f_1_s_2_3'),('f_1_s_2_3','f_1_s_3_3'),('f_1_s_3_3','f_1_s_4_3'),('f_1_s_4_3','f_1_s_5_3'),('f_1_s_5_3','f_1_s_6_3'),('f_1_s_6_3','f_1_s_7_3'),('f_1_s_7_3','f_1_s_8_3'),('f_1_s_0_3','f_1_s_1_3'),('f_1_s_8_3','f_1_s_9_3'),('f_1_s_0_3','f_1_s_0_4'),('f_1_s_1_3','f_1_s_1_4'),('f_1_s_2_3','f_1_s_2_4'),('f_1_s_3_3','f_1_s_3_4'),('f_1_s_4_3','f_1_s_4_4'),('f_1_s_5_3','f_1_s_5_4'),('f_1_s_6_3','f_1_s_6_4'),('f_1_s_7_3','f_1_s_7_4'),('f_1_s_8_3','f_1_s_8_4'),('f_1_s_9_3','f_1_s_9_4'),('s_3_20','f_1_s_3_20'),('s_3_21','f_1_s_3_21')]
    for state in S:
        T.append((state,state))
    AP = ['E1','E2','K','P','M1','M2','O1','O2','O3','O4','O5','O6','O7','F1_E1','F1_E2','F1_K','F1_P','F1_M1','F1_M2','F1_O1','F1_O2','F1_O3','F1_O4','F1_O5','F1_O6','F1_O7']
    
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
        C[('f_1_s_8_1',t,'f_1_s_8_2')] = 2
        C[('f_1_s_8_2',t,'f_1_s_8_3')] = 2
        C[('f_1_s_5_1',t,'f_1_s_5_2')] = 3
        C[('f_1_s_5_2',t,'f_1_s_5_3')] = 3
        C[('f_1_s_8_2',t,'f_1_s_8_1')] = 2
        C[('f_1_s_8_3',t,'f_1_s_8_2')] = 2
        C[('f_1_s_5_2',t,'f_1_s_5_1')] = 3
        C[('f_1_s_5_3',t,'f_1_s_5_2')] = 3
        C[('f_1_s_3_21',t,'f_1_s_3_3')] = 3
        C[('f_1_s_3_3',t,'f_1_s_3_21')] = 3
        C[('f_1_s_3_20',t,'f_1_s_3_1')] = 3
        C[('f_1_s_3_1',t,'f_1_s_3_20')] = 3
        C[('s_3_20',t,'f_1_s_3_20')] = 3
        C[('f_1_s_3_20',t,'s_3_20')] = 2
        C[('s_3_21',t,'f_1_s_3_21')] = 3
        C[('f_1_s_3_21',t,'s_3_21')] = 2


    
    L = {'s_3_20':'E1','s_3_21':'E2','s_9_2':'K','s_5_2':'P','s_0_2':'M1','s_3_4':'M2','s_2_0':'O1','s_4_0':'O2','s_8_0':'O3','s_2_4':'O4','s_4_4':'O5','s_9_1':'O6','s_6_0':'O7','f_1_s_3_20':'F1_E1','f_1_s_3_21':'F1_E2','f_1_s_9_2':'F1_K','f_1_s_5_2':'F1_P','f_1_s_0_2':'F1_M1','f_1_s_3_4':'F1_M2','f_1_s_2_0':'F1_O1','f_1_s_4_0':'F1_O2','f_1_s_8_0':'F1_O3','f_1_s_2_4':'F1_O4','f_1_s_4_4':'F1_O5','f_1_s_9_1':'F1_O6','f_1_s_6_0':'F1_O7'}
    wts = WTS(S,s_0,T,AP,L,C)
    
    

    #Eventually_[4,7] x=1,y=8
    e1 = MTLFormula.Eventually(MTLFormula.Predicate('K'),6,12)
    e2 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),115,118)
    a1 = MTLFormula.Always(MTLFormula.Predicate('E2'),300,305)
    e3 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),400,405)
    e4 = MTLFormula.Eventually(MTLFormula.Predicate('F1_O1'),407,409)
    
    a2 = MTLFormula.Always(MTLFormula.Predicate('P'),250,260)
    a3 = MTLFormula.Always(MTLFormula.Predicate('E1'),320,346)
    e5 = MTLFormula.Eventually(MTLFormula.Predicate('O2'),390,410)
    e6 = MTLFormula.Eventually(MTLFormula.Predicate('F1_M1'),400,420)
    e7 = MTLFormula.Eventually(MTLFormula.Predicate('F1_E2'),455,459)
    
    n1 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('O3'),130,145))
    n2 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('K'),150,165))
    n3 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('M1'),110,128))
    n4 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('F1_O4'),12,25))
    c1 = MTLFormula.Conjunction([MTLFormula.Always(MTLFormula.Predicate('O5'),310,315),MTLFormula.Always(MTLFormula.Predicate('O2'),376,380)]) 
    c2 = MTLFormula.Conjunction([MTLFormula.Eventually(MTLFormula.Predicate('O6'),210,240),MTLFormula.Always(MTLFormula.Predicate('F1_O4'),40,50)]) 
    d1 = MTLFormula.Disjunction([MTLFormula.Eventually(MTLFormula.Predicate('F1_K'),410,440),MTLFormula.Eventually(MTLFormula.Predicate('O5'),460,470)]) 
    d2 = MTLFormula.Disjunction([MTLFormula.Eventually(MTLFormula.Predicate('M2'),221,249),MTLFormula.Eventually(MTLFormula.Predicate('F1_E1'),111,149)]) 
    ae = MTLFormula.Always(MTLFormula.Eventually(MTLFormula.Predicate('F1_O1'),5,10),150,170)
    ea = MTLFormula.Eventually(MTLFormula.Always(MTLFormula.Predicate('O7'),5,10),340,360)
    
    
    Demands = [(e1,1),(e2,1),(a1,1),(e3,3),(e4,1),(a2,1),(a3,3),(e5,3),(e6,2),(e7,1),(n1,1),(n2,3),(n3,2),(n4,1),(c1,1),(c2,2),(d1,1),(d2,3),(ea,2),(ae,2)]
    

    trajectory = generate_plan_mtl_wts(Demands,wts,PLANNING_HORIZON)
    
    print(trajectory)


