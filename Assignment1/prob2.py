import numpy as np

def calculate_prob(M,k):
    # variable to go through columns where it has the value k
    selected_column = 0
    # create empty matrix to calculate probability
    ep = [[0 for i in range(n)]  
             for i in range(m)]
    # create empty vector to add the total value
    total = [0 for i in range(n)]
    
    # get the column
    # if you encounter the value in certain column, stick to it
    for i in range(0,n):
        for j in range(0,n):
            if M[i][j]==k:
                selected_column = j
        break
    
    # calculate first row(index = 0)
    # only columns are moving
    # first row does not have be updated
    for j in range(0,n):
        ep[0][j] = M[0][j]
        total[0]+=ep[0][j]
    
    # starts from second row(index = 1)
    for i in range(1,n):
        for j in range(m):
            # set element and divide ep with total of the row
            ep[i][j] += ep[i-1][j]/total[i-1]
            # add calculated ep to the same column of next row
            ep[i][j] += M[i][j]
            # in order to update the total of the row
            total[i] += ep[i][j]
    
    result = ep[n-1][selected_column] / total[n-1]
    # return result
    return round(result,3)

# test sample  
#M= [[0,1],[1,1]] 
M = [[2,3],[1,4]]
k = 1
# row
n = len(M)
# column
m = len(M[0])
# call the method to get probability
print(calculate_prob(M,k))