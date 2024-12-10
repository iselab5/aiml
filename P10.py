import numpy as np
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
print("small x",x)
#original output 
y = np.array(([92], [86], [89]), dtype=float)
X = x/np.amax(x,axis=0) #maximum along the first axis 
print("Capital X",X)
#Defining Sigmoid Function for output 
def sigmoid (x):
    return (1/(1 + np.exp(-x)))
#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)
#Variables initialization
epoch=7000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = 2 #number of input layer neurons 
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer
#Defining weight and biases for hidden and output layer 
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
#Forward Propagation 
for i in range(epoch):
    hinp1=np.dot(X,wh) 
    hinp=hinp1 + bh
    hlayer_act = sigmoid(hinp)
    outinp1=np.dot(hlayer_act,wout)
    outinp = outinp1+ bout
    output = sigmoid(outinp)

#Backpropagation Algorithm 
EO = y-output
outgrad = derivatives_sigmoid(output)
d_output = EO* outgrad
EH = d_output.dot(wout.T)
hiddengrad = derivatives_sigmoid(hlayer_act)
#how much hidden layer wts contributed to error
d_hiddenlayer = EH * hiddengrad
wout += hlayer_act.T.dot(d_output) *lr
# dotproduct of nextlayererror and currentlayerop
bout += np.sum(d_output, axis=0,keepdims=True) *lr
#Updating Weights
wh += X.T.dot(d_hiddenlayer) *lr
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)