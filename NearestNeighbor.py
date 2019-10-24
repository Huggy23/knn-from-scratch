"""
CPSC-51100, SUMMER 2019
NAME: JASON HUGGY
PROGRAMMING ASSIGNMENT #3
"""

import numpy as np

train_values = np.loadtxt("iris-training-data.csv", usecols = (0,1,2,3), 
                          delimiter=",") # inlcludes numerical values only
train_labels = np.loadtxt("iris-training-data.csv", usecols = (4), 
                          delimiter=",", dtype= object) # labels only
test_values = np.loadtxt("iris-testing-data.csv", usecols = (0,1,2,3), 
                         delimiter=",")
test_labels = np.loadtxt("iris-testing-data.csv", usecols = (4), 
                         delimiter=",", dtype=object)

# this function will take the training and testing values and compute the 
# distance formula between every single pair to find the closest index
# the closest point's label is returned for each set of testing values
def nearest_neighbor(train_values, train_labels, test_values):
    predicted_labels = np.array([])
    
    # for each characteristic in test
    for i, j, k, l in test_values:  
        result = np.array([])
        
        # for each characteristic in train
        for x, y, z, q in train_values: 
            
            # the distance formula applied
            result= np.append(result,(np.sqrt((i - x)**2 + (j - y)**2 + 
                                              (k - z)**2 + (l - q)**2)))
            
        #returns index of closest point
        nearest_index = (np.argmin(result)) 
        
        #assigns label from same index in training data
        predicted_labels = np.append(predicted_labels,
                                     (train_labels[nearest_index]))
                
        
    return predicted_labels

predicted = nearest_neighbor(train_values, train_labels, test_values)
print('#, True, Predicted')

index = range(1, len(test_labels)+1)

#lists the true value and what was predicted for each set of values
for i, j, k in zip(index, test_labels, predicted):
    print(str(i) + ','+ j +','+ k)
    
# calculates the accuracy of the model    
def accuracy(test_labels, predicted):
    count = 0
    
    for i,j in zip(test_labels, predicted):
    
        if i == j:
            count+=1 
        
    accuracy = (count / len(test_labels)) * 100
    print('Accuracy:', '%.2f'%accuracy+'%')
    
accuracy(test_labels, predicted)