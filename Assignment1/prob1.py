import numpy as np

def calcProbability(p,n):
    # creating empty array
    ep = np.zeros((n+1,n+1))
    
    # baseline
    ep[0][0] = 1.0
    
    # sum up
    result = 0.0
    
    # calculate probability for each case
    # a = number of coins, b = number of heads in ath coins
    for a in range(1, n+1):
        for b in range(a+1):
            # when there is no head
            if (b==0):
                ep[a][b] = ep[a-1][b]*(1.0-p[a-1])
            else:
                ep[a][b] = ep[a-1][b]*(1.0-p[a-1]) + ep[a-1][b-1]*p[a-1]    
            #print(ep[a][b])
                
    
    # get result
    # if the number of heads is greater than the half
    for i in range((n+1)//2,n+1):
        result+=ep[n][i]
        
    return round(result,3)

# test input
# define probability of getting heads in each coin
x = [0.3,0.4,0.7]
# number of coins
n = len(x)
print(calcProbability(x,n))