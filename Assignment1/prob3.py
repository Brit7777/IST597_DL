import numpy as np

# method to mutiply two matrices M and N given the number of states n
def multiplyM(M,N,n):
    #result_matrix = np.zeros((n,n), dtype=np.float64)
    result_matrix = [[0 for i in range(n)]  
                        for i in range(n)]
    for i in range(0,n): 
        for j in range(0,n): 
            for k in range(0,n):
                result_matrix[i][j] += M[i][k] * N[k][j]
    return result_matrix

# method to calculate te power of a matrix
def powerM(M, T, n):
    power_matrix = [[0 for i in range(n)]  
                        for i in range(n)]
    # creating indentity matrix
    for i in range(0,n):
        power_matrix[i][i] = 1
    
    while (T):
        if (T%2)==0:
            power_matrix = multiplyM(power_matrix,M,n)
        M = multiplyM(M,M,n)
        T /= 2
    
    return power_matrix
        
        
    
# method to calculate the probability of  
# reaching F at time T after starting from S 
def calcProbability(M,S,F,T,n):
    prob_matrix = powerM(M,T,n)
    return prob_matrix[F-1][S-1]



# each row must add up to exactly 1
# row is the start, column is the end
M = [[0.0, 0.23, 0.0, 0.77,0.0,0.0],
    [0.09, 0.0, 0.06, 0.0, 0.0, 0.85],
    [0.0, 0.0, 0.0, 0.63, 0.0, 0.37],
    [0.0, 0.0, 0.0, 0.0, 0.65, 0.35],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.62, 0.0, 0.0, 0.38, 0.0]]

# sample test
# S stands for starting state
S = 4
# F stands for finishing state
F = 6
# T stands for finishing time
T = 20
# n stands for number of states
n = len(M)

# get result
print(calcProbability(M,S,F,T,n))