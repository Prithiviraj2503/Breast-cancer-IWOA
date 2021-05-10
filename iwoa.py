"""
Created on Fri Feb 19 21:58:37 2021
Project Title: Feature Selection Using Improved Whale Optimization Algorithm for High Dimensional Microarray Data

@Author: Prithiviraj K
Reg. No: 810017205062
Final year-IT-'B'.

@Guided By: Dr. S. Sathiya Devi
Department of Information Technology
University College of Engineering, BIT campus- Tiruchirappalli.
"""

import numpy as np
from numpy.random import rand
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#from sklearn.ensemble import GradientBoostingClassifier


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    return x

def error_rate(xtrain, ytrain, x, opts):
    # parameters
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']
    
    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug   
    # Training
    #gb= xgb.XGBClassifier(random_state=1, learning_rate=0.1)
    model = SVC(kernel="linear")
    model.fit(xtrain, ytrain)
    #Prediction 
    ypred    = model.predict(xvalid)

    acc     = accuracy_score(yvalid,ypred)
    error   = 1 - acc
    perf_dict={'accuracy':acc, 'error_rate':error}
    return perf_dict

# Error rate & Feature size
feat=[]
def Fun(xtrain, ytrain, x, opts):
    # Parameters
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    print("Selected Features Size", num_feat)
    feat.append(num_feat)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
        findict={'accuracy':0, 'error_rate':1, 'cost':cost , 'selfeat':feat}
    else:
        per_dict = error_rate(xtrain, ytrain, x, opts)# Get error rate
        error=per_dict['error_rate']
        cost  = alpha * error + beta * (num_feat / max_feat)# Objective function
        accuracy=per_dict['accuracy']
        findict={'accuracy':accuracy, 'error_rate':error, 'cost':cost , 'selfeat':feat}
    return findict

maxacclist=[]
minerrlist=[]
def best(ac_list, er_list):
    maxacc=max(ac_list)
    maxacclist.append(maxacc)
    minerr=min(er_list)
    minerrlist.append(minerr)
    dictfit={'ac':maxacclist,'er':minerrlist}
    return dictfit

def mutation(parent):
    random_value = np.random.uniform(-1.0, 1.0, 1)
    parent = parent + random_value
    return parent

def crossover(p1,p2):
    crosspoint=np.random.rand()
    pc1=p1%crosspoint
    pc2=p2%crosspoint
    offspring1= pc1+p2
    offspring2= pc2+p1
    offspring=offspring1 and offspring2
    return offspring
    

def woa(xtrain, ytrain, opts):
    ub    = 1
    lb    = 0
    thres = 0.5
    b     = 1   
    # constant    
    N        = opts['N']
    max_iter = opts['T']
    if 'b' in opts:
        b    = opts['b']
    
    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
        
    # Initialize position 
    X    = init_position(lb, ub, N, dim)
    
    # Binary conversion
    Xbin = binary_conversion(X, thres, N, dim)
    
    # Fitness at first iteration
    fit  = np.zeros([N, 1], dtype='float')
    Xgb  = np.zeros([1, dim], dtype='float')
    fitG = float('inf')
    
    print("Random Solution Generation:")
    for i in range(N):
        print("Whale -> ",i+1)
        
        funCall=Fun(xtrain, ytrain, Xbin[i,:], opts)
        fit[i,0]=funCall['cost']
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
        
    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    
    t = 0  
    curve[0,t] = fitG.copy()
    print("Best (WOA):", curve[0,t])
    #print("\nGeneration:", t + 1)
    #t += 1

    while t < max_iter:
        print("Generation:", t)
        # Define a, linearly decreases from 2 to 0 
        a = 2 - t * (2 / max_iter)
        
        for i in range(N):
            # Parameter A (2.3)
            A = 2 * a * rand() - a
            # Paramater C (2.4)
            # Parameter p, random number in [0,1]
            p = rand()
            # Parameter l, random number in [-1,1]
            l = -1 + 2 * rand()  
            # Whale position update (2.6)
            if p  < 0.5:
                # {1} Encircling prey
                if abs(A) < 1:
                    for d in range(dim):
                       # Dx = abs(C * Xgb[0,d] - X[i,d])
                       # X[i,d] = Xgb[0,d] - A * Dx
                        X[i,d]=crossover(Xgb[0,d], X[i,d])
                        X[i,d]=mutation(X[i,d])
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                
                # {2} Search for prey
                elif abs(A) >= 1:
                    for d in range(dim):
                        # Select a random whale
                        k      = np.random.randint(low = 0, high = N)
                        #Dx     = abs(C * X[k,d] - X[i,d])
                        #X[i,d] = X[k,d] - A * Dx
                        X[i,d]= crossover(X[k,d],X[i,d])
                        X[i,d]= mutation(X[k,d])
                        X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
                        
            
            # {3} Bubble-net attacking 
            elif p >= 0.5:
                for d in range(dim):
                    # Distance of whale to prey
                    dist   = abs(Xgb[0,d] - X[i,d])
                    # Position update (2.5)
                    X[i,d] = dist * np.exp(b * l) * np.cos(2 * np.pi * l) + Xgb[0,d] 
                    # Boundary
                    X[i,d] = boundary(X[i,d], lb[0,d], ub[0,d])
        
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        accuracylist=[]
        errorlist=[]
        for i in range(N):
            funCall=Fun(xtrain, ytrain, Xbin[i,:], opts)
            fit[i,0]=funCall['cost']
            accuracy=funCall['accuracy']
            #selfeat=funCall['selfeat']
            error=funCall['error_rate']
            print("Accuracy",accuracy,"error",error) 
            accuracylist.append("%.6f"%accuracy)
            errorlist.append("%.6f"%error)
        
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]

        #Store result
        curve[0,t] = fitG.copy()
        #print("Accuracy List:",accuracylist)
        #print("Error List",errorlist)
        print("Best Whale solution:", curve[0,t],"\n")
        bestfit=best(accuracylist,errorlist)
        t += 1      
    
   # print("Selected Features",selfeat)           
    #Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)    
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    bestac=bestfit['ac']
    bester=bestfit['er']
    print("\n Best Accury gained over entire iteration", bestac)
    print("Minimized Error rate gained over entire iteration", bester)
    print("Indexes of Selected features",sel_index)
    print("Feature size", num_feat)
    
    
    # Create dictionary
    woa_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return woa_data 
