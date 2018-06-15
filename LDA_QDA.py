import numpy as np

from sympy import *


import matplotlib.pyplot as plt

x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)

label= np.ones_like(x)

label[0:100]=0

plt.scatter(x,y,c=label)


def LDA():
    
    Mu1 = [np.mean(x[0:100]),np.mean(y[0:100])]
    
    Mu2 = [np.mean(x[100:200]),np.mean(y[100:200])]
    
    X = ['x1', 'x2']
    
    SigmaInverse = np.linalg.inv(np.cov(x,y))
#    print(SigmaInverse)
    x1, x2 = symbols('x1 x2')
    X = [x1, x2]
    
    X_Mu1 = [poly(X[0] - Mu1[0]), poly(X[1] - Mu1[1])]
    X_Mu1 = np.reshape(X_Mu1,(1,2))
    #print(X_Mu1)
    X_Mu1_Transpose = np.transpose(X_Mu1)
    
    X_Mu2 = [poly(X[0] - Mu2[0]), poly(X[1] - Mu2[1])]
    X_Mu2 = np.reshape(X_Mu2,(1,2))
    #print(X_Mu2)
    X_Mu2_Transpose = np.transpose(X_Mu2)
    
    LHS = np.dot(X_Mu1,SigmaInverse)
    LHS = np.dot(LHS,X_Mu1_Transpose)
    
    RHS = np.dot(X_Mu2,SigmaInverse)
    RHS = np.dot(RHS,X_Mu2_Transpose)
    
    Resultant_Eq = LHS - RHS
    #print("Resultant_Eq_LDA ",Resultant_Eq,"\n")
    
    expr = Resultant_Eq[0,0]
    print("Resultant_Equation_LDA : ",expr,'\n')
    a = []
    b = []
    
    expr_x1 = solve([expr],[x1])
    #print(expr_x1)
    Final_x1 = expr_x1[x1].subs(x2,0)
    a.append(Final_x1)
    b.append(0)
    
    expr_x2 = solve([expr],[x2])
    Final_x2 = expr_x2[x2].subs(x1,0)
    b.append(Final_x2)
    a.append(0)
    expr_x2 = solve([expr],[x2])
    Final_x2 = expr_x2[x2].subs(x1,-1)
    b.append(Final_x2)
    a.append(-1)
    
    plt.plot(a,b)
    plt.show

LDA()



def QDA():
    
    Mu1_QDA = [np.mean(x[0:100]),np.mean(y[0:100])]  
    Mu2_QDA = [np.mean(x[100:200]),np.mean(y[100:200])]

    X_One_Hun = x[0:100]
    Y_One_Hun = y[0:100]
    
    X_Hun_Two = x[100:200]
    Y_Hun_Two = y[100:200]
    
    SigmaInverse_1_QDA = np.linalg.inv(np.cov(X_One_Hun,Y_One_Hun))
    #print(SigmaInverse_1_QDA)
    SigmaInverse_2_QDA = np.linalg.inv(np.cov(X_Hun_Two,Y_Hun_Two))
    #print(SigmaInverse_2_QDA)

    X_Arr = ['xx', 'yy']
    
    xx, yy = symbols('xx yy')
    X_Arr = [xx, yy]
    
    X_Mu1_QDA = [poly(X_Arr[0] - Mu1_QDA[0]), poly(X_Arr[1] - Mu1_QDA[1])]
    X_Mu1_QDA = np.reshape(X_Mu1_QDA,(1,2))
    
    X_Mu1_QDATranspose = np.transpose(X_Mu1_QDA)
    
    X_Mu2_QDA = [poly(X_Arr[0] - Mu2_QDA[0]), poly(X_Arr[1] - Mu2_QDA[1])]
    #X_Mu2_QDA = np.reshape(Mu2_QDA,(1,2))
    
    
    X_Mu2_QDATranspose = np.transpose(X_Mu2_QDA)
    
    LHS_QDA = np.dot(X_Mu1_QDA,SigmaInverse_1_QDA)
    LHS_QDA = np.dot(LHS_QDA,X_Mu1_QDATranspose)
    LHS_QDA = LHS_QDA[0,0]
    
    RHS_QDA = np.dot(X_Mu2_QDA,SigmaInverse_2_QDA)
    RHS_QDA = np.dot(RHS_QDA,X_Mu2_QDATranspose)   
    
    Resultant_Eq_QDA = LHS_QDA - RHS_QDA
    print('Resultant_Equation_QDA : ',Resultant_Eq_QDA)
    
    X_Values = np.linspace(-4,4,30)
    Y_Values = np.linspace(-4,4,30)
    
    X_Grid,Y_Grid = np.meshgrid(X_Values, Y_Values)

    Mesh_Array = []
    for i in range(0,30):
        Final_QDA = []
        for j in range(0,30):
            Final_QDA.append(Resultant_Eq_QDA.subs({xx:X_Grid[i][j] ,yy:Y_Grid[i][j] }))
        Mesh_Array.append(Final_QDA)
    #print(np.shape(Mesh_Array))
    
    plt.contour(X_Grid,Y_Grid,Mesh_Array,1)
    plt.show()
    
QDA()