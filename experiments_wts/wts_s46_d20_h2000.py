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
    PLANNING_HORIZON = 2000

    #The WTS of LV24, floor 4
    S = ['s_0_0','s_1_0','s_2_0','s_3_0','s_4_0','s_5_0','s_6_0','s_7_0','s_8_0','s_0_1','s_1_1','s_2_1','s_3_1','s_4_1','s_5_1','s_6_1','s_7_1','s_8_1','s_9_1','s_0_2','s_1_2','s_3_20','s_3_21','s_5_2','s_8_2','s_9_2','s_0_3','s_1_3','s_2_3','s_3_3','s_4_3','s_5_3','s_6_3','s_7_3','s_8_3','s_9_3','s_0_4','s_1_4','s_2_4','s_3_4','s_4_4','s_5_4','s_6_4','s_7_4','s_8_4','s_9_4']
    s_0 = 's_0_3'
    
    T = [('s_0_0','s_0_1'),('s_1_0','s_1_1'),('s_2_0','s_2_1'),('s_3_0','s_3_1'),('s_4_0','s_4_1'),('s_5_0','s_5_1'),('s_6_0','s_6_1'),('s_7_0','s_7_1'),('s_8_0','s_8_1'),('s_0_1','s_1_1'),('s_1_1','s_2_1'),('s_2_1','s_3_1'),('s_3_1','s_4_1'),('s_4_1','s_5_1'),('s_5_1','s_6_1'),('s_6_1','s_7_1'),('s_7_1','s_8_1'),('s_8_1','s_9_1'),('s_1_1','s_1_2'),('s_0_2','s_1_2'),('s_1_2','s_1_3'),('s_3_1','s_3_20'),('s_3_21','s_3_3'),('s_5_1','s_5_2'),('s_5_2','s_5_3'),('s_8_1','s_8_2'),('s_8_2','s_9_2'),('s_8_2','s_8_3'),('s_0_3','s_1_3'),('s_1_3','s_2_3'),('s_2_3','s_3_3'),('s_3_3','s_4_3'),('s_4_3','s_5_3'),('s_5_3','s_6_3'),('s_6_3','s_7_3'),('s_7_3','s_8_3'),('s_0_3','s_1_3'),('s_8_3','s_9_3'),('s_0_3','s_0_4'),('s_1_3','s_1_4'),('s_2_3','s_2_4'),('s_3_3','s_3_4'),('s_4_3','s_4_4'),('s_5_3','s_5_4'),('s_6_3','s_6_4'),('s_7_3','s_7_4'),('s_8_3','s_8_4'),('s_9_3','s_9_4')]
    for state in S:
        T.append((state,state))
    AP = ['E1','E2','K','P','M1','M2','M1','Jana','Alexis','O1','O2','O3','O4','O5']
    
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


    
    L = {'s_3_20':'E1','s_3_21':'E2','s_9_2':'K','s_5_2':'P','s_0_2':'M1','s_3_4':'M2','s_9_1':'Jana','s_6_0':'Alexis','s_0_3':'Base','s_2_0':'O1','s_4_0':'O2','s_8_0':'O3','s_2_4':'O4','s_4_4':'O5'}
    wts = WTS(S,s_0,T,AP,L,C)
    
    

    #Eventually_[4,7] x=1,y=8
    e1 = MTLFormula.Eventually(MTLFormula.Predicate('K'),6,12)
    e2 = MTLFormula.Eventually(MTLFormula.Predicate('M2'),15,18)
    a1 = MTLFormula.Always(MTLFormula.Predicate('E2'),400,500)
    e4 = MTLFormula.Eventually(MTLFormula.Predicate('Jana'),610,650)
    a2 = MTLFormula.Always(MTLFormula.Predicate('Alexis'),1600,1800)
    
    a2 = MTLFormula.Always(MTLFormula.Predicate('P'),310,330)
    a3 = MTLFormula.Always(MTLFormula.Predicate('E1'),370,400)
    e5 = MTLFormula.Eventually(MTLFormula.Predicate('Alexis'),420,430)
    e6 = MTLFormula.Eventually(MTLFormula.Predicate('M1'),1100,1110)
    e7 = MTLFormula.Eventually(MTLFormula.Predicate('E2'),1410,1490)
    
    n1 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('Base'),1130,1145))
    n2 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('K'),1115,1145))
    n3 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('M1'),1110,1128))
    n4 = MTLFormula.Negation(MTLFormula.Always(MTLFormula.Predicate('O2'),1111,1125))
    c1 = MTLFormula.Conjunction([MTLFormula.Always(MTLFormula.Predicate('O1'),1520,1525),MTLFormula.Always(MTLFormula.Predicate('O2'),1626,1630)]) 
    c2 = MTLFormula.Conjunction([MTLFormula.Eventually(MTLFormula.Predicate('O3'),1410,1440),MTLFormula.Always(MTLFormula.Predicate('O4'),1840,1842)]) 
    d1 = MTLFormula.Disjunction([MTLFormula.Eventually(MTLFormula.Predicate('K'),1910,1940),MTLFormula.Eventually(MTLFormula.Predicate('O5'),1940,1949)]) 
    d2 = MTLFormula.Disjunction([MTLFormula.Eventually(MTLFormula.Predicate('M2'),1,49),MTLFormula.Eventually(MTLFormula.Predicate('E1'),1,49)]) 
    ae = MTLFormula.Always(MTLFormula.Eventually(MTLFormula.Predicate('O1'),5,10),1350,1420)
    ea = MTLFormula.Eventually(MTLFormula.Always(MTLFormula.Predicate('O3'),5,10),1550,1590)
    

    Demands = [(e1,1),(e2,1),(a1,1),(e3,3),(e4,1),(a2,1),(a3,3),(e5,3),(e6,2),(e7,1),(n1,1),(n2,3),(n3,2),(n4,1),(c1,1),(c2,2),(d1,1),(d2,3),(ea,2),(ae,2)]

    trajectory = generate_plan_mtl_wts(Demands,wts,PLANNING_HORIZON)
    
    print(trajectory)


