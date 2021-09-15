# -*- coding: utf-8 -*-

# Initialize the LR, bias and weights
lr = 1
bias = 1
weights = [-50, 20, 30]


# Func Perceptron - input and output - adjust the weights - training the model

def perceptron(x_1, x_2, output_A):
    output_P = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    #Apply activation function
    if output_P > 4.6:
        output_P = 1
    else:
        output_P = 0
    #Calculate Error    
    error = 1/2*(output_P - output_A)**2
    #adjust the weights
    weights[0] = weights[0] + error*bias*lr
    weights[1] = weights[1] + error*x_1*lr
    weights[2] = weights[2] + error*x_2*lr
    

# Func Predict function
def predict(x_1, x_2):
    output_P = bias*weights[0] + x_1*weights[1] + x_2*weights[2]
    #Apply activation function
    if output_P > 4.6:
        output_P = 1
    else:
        output_P = 0

    return output_P

# run multiple times
for i in range(40):
    #training data
    perceptron(0,0,0)
    perceptron(0,1,1)
    perceptron(1,0,1)
    perceptron(1,1,1)
    print(weights)
    
x_1 = int(input("ENter first input"))
x_2 = int(input("Enter second input"))

pred = predict(x_1, x_2)
print(x_1, "or", x_2, "---->", pred)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
