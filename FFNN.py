import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mnist_train.csv")

#labels / expected values
labels = np.array(data["label"])
targetArray = np.zeros((labels.size,10))

#sigmoid function to calculate activation
def sigmoid(x):
    newArray = np.zeros((x.size,1))
    for i in range(0,x.size):
        newArray[i]=1/(1+np.exp(-x[i]))
    return newArray

#from textbook
def sigPrime(x):
    return(sigmoid(x))*(1-sigmoid(x))

sizehid = 12
sizeout=10

################ LAYERS
#Input Layer
inputArray = np.array(data.iloc[:,1:785])
rowsin, sizein = inputArray.shape
#into 0-1 activation
inputArray = inputArray/255

######## WEIGHTS AND BIASES
w01 = np.random.randint(-1000,high=1000, size=(sizein,sizehid))
w01 = w01/1000

b1 = np.random.randint(-10,high=10,size=sizehid)[np.newaxis]
b1=b1.T
b1 = b1/27

w12 = np.random.randint(-1000,high=1000, size=(sizehid,sizehid))
w12 = w12/1000

b2 = np.random.randint(-10,high=10,size=sizehid)[np.newaxis]
b2=b2.T
b2=b2/27

w23 = np.random.randint(-1000,high=1000, size=(sizehid,sizeout))
w23 = w23/1000

b3 = np.random.randint(-10,high=10,size=sizeout)[np.newaxis]
b3=b3.T
b3=b3/27

#layer activations
layer1 = np.zeros((12,1))
layer2 = np.zeros((12,1))
outputlayer = np.zeros((10,1))

#layers pre activation
prelayer1 = np.zeros((12,1))
prelayer2 = np.zeros((12,1))
preoutputlayer = np.zeros((10,1))

#error in layers
layer1error = np.zeros((12,1))
layer2error = np.zeros((12,1))
outputlayererror = np.zeros((10,1))

#error in bias
l1biaserror = np.zeros(layer1.shape)
l2biaserror = np.zeros(layer2.shape)
outputbiaserror = np.zeros(outputlayer.shape)

#error in weights
l1weighterror = np.zeros(w01.shape)
l2weighterror = np.zeros(w12.shape)
outputweighterror = np.zeros(w23.shape)

#cost
costFns = 0

def softmax(x):
    newArray = np.exp(x)
    sum = np.sum(newArray)
    for i in range(0,newArray.size):
        newArray[i]=newArray[i]/sum
    return newArray

#initialize target array values
def initArray(l,z,size):
    for i in range(0,size):
        number = l[i]
        z[i][number]=1

def linFn(weights,neurons,bias):
    return(weights.T @ neurons)+bias

#Cost function for a single input-output pair (output = activations vector, input  = target vector)
def singleCostFn(output,desired):
    global costFns
    res = -1*(np.sum(np.multiply(desired,np.log(output))))
    costFns+=res
    return 1

#input multiple cost functions, take mean, store as cost function
def multipleCostFn(x):
    global costFns
    costFns/=x
    return 1

#change in cost wrt any value neuron in the output, make sure all vertical vectors
def outputerror(output, desired):
    return (output - desired)

#change in cost wrt any value neuron previous to output layer (minus input)
def layererror(prevweight,errorprev,preoutput):
    return (np.multiply(prevweight @ errorprev,sigPrime(preoutput)))

#change in cost wrt any bias in any layer
def biaserror():
    global l1biaserror
    global l2biaserror
    global outputbiaserror

    l1biaserror += layer1error
    l2biaserror += layer2error
    outputbiaserror += outputlayererror
    return 1

#change in cost wrt any weight between a layer and it's previous layer's activations
def weighterror(x):
    global l1weighterror
    global l2weighterror
    global outputweighterror

    outputweighterror += layer2 @ outputlayererror.T
    l2weighterror += layer1 @ layer2error.T
    l1weighterror += inputArray[x][np.newaxis].T @ layer1error.T
    return 1

def feedforward(inputlayer):
    #weights are input rows by output columns
    #input layer and bias are vertical vectors
    global layer1
    global layer2
    global outputlayer

    global prelayer1
    global prelayer2
    global preoutputlayer

    prelayer1 = linFn(w01,inputlayer,b1)
    layer1 = sigmoid(prelayer1)
    prelayer2 = linFn(w12,layer1,b2)
    layer2 = sigmoid(prelayer2)
    preoutputlayer = linFn(w23,layer2,b3)
    outputlayer = softmax(preoutputlayer)
    return outputlayer

#calculate the error in each neuron for a target array with x
def calcError(target,x):
    global layer1
    global layer2
    global outputlayer

    global layer1error
    global layer2error
    global outputlayererror

    outputlayererror = outputerror(outputlayer,target[x][np.newaxis].T)
    layer2error = layererror(w23,outputlayererror,prelayer2)
    layer1error = layererror(w12,layer2error,prelayer1)
    return 1

def backpropogate():
    global w01
    global w12
    global w23

    global l1weighterror
    global l2weighterror
    global outputweighterror

    #grad descent
    w01= w01-2*l1weighterror
    w12= w12-2*l2weighterror
    w23= w23-2*outputweighterror

    #reset weight error
    l1weighterror = np.zeros(w01.shape)
    l2weighterror = np.zeros(w12.shape)
    outputweighterror = np.zeros(w23.shape)

    global b1
    global b2
    global b3

    global l1biaserror
    global l2biaserror
    global outputbiaserror

    b1= b1-2*l1biaserror
    b2= b2-2*l2biaserror
    b3= b3-2*outputbiaserror

    l1biaserror = np.zeros(b1.shape)
    l2biaserror = np.zeros(b2.shape)
    outputbiaserror = np.zeros(b3.shape)
    return 1

def meanerror(x):
    global l1weighterror
    global l2weighterror
    global outputweighterror

    global l1biaserror
    global l2biaserror
    global outputbiaserror

    l1weighterror/=x
    l2weighterror/=x
    outputweighterror/=x
    l1biaserror/=x
    l2biaserror/=x
    outputbiaserror/=x
    return 1

def predict(output):
    return np.argmax(output)

##############################
###    MAIN CODE
##############################

#TEST
#initializing training array, contains flattened 28x28 grid of target data
initArray(labels,targetArray,60000)

#Mini batch and epoch size
minibatch = 60
epochs = 1000

#Cost Function and accuracy plots
plting=[]
plting2 =[]

for j in range(0,epochs):
    count=0
    costFns = 0
    for i in range(0,minibatch):
        index = i+j*minibatch
        feedforward(inputArray[index][np.newaxis].T)
        singleCostFn(outputlayer,targetArray[index][np.newaxis].T)
        if(predict(outputlayer)==labels[index]):
            count+=1
        #current layer error
        calcError(targetArray,index)

        #add to total bias and weight error
        biaserror()
        weighterror(index)

    meanerror(minibatch)
    multipleCostFn(minibatch)
    backpropogate()
    plting2.append(count/minibatch)
    plting.append(costFns)

plt.subplot(121)
plt.plot(plting)
plt.xlabel("Epochs", fontsize = 12)
plt.ylabel('Cost Function', fontsize = 12)
plt.title('Cost Function over 1000 Epochs', fontsize = 16)
plt.subplot(122)
plt.plot(plting2)
plt.xlabel("Epochs", fontsize = 12)
plt.ylabel('Accuracy', fontsize = 12)
plt.title('Average System Accuracy over 1000 Epochs', fontsize = 16)
plt.show()
#back propogation
#output to first hidden layer

data2 = pd.read_csv("mnist_test.csv")

#labels / expected values
labels2 = np.array(data2["label"])
targetArray2 = np.zeros((labels2.size,10))

#Input Layer
inputArray2 = np.array(data2.iloc[:,1:785])
rowsin2, sizein2 = inputArray2.shape
#into 0-1 activation
inputArray2 = inputArray2/255

count=0


#TEST
for i in range(0,10000):
    feedforward(inputArray2[i][np.newaxis].T)
    if(predict(outputlayer)==labels2[i]):
        count+=1

print(count/10000)


dp = np.array([0.8909, 0.8904, 0.896, 0.893, 0.9021, 0.8729, 0.8869, 0.8909, 0.8905, 0.8802])

mean = np.mean(dp)  # Mean of the distribution
std_dev = np.std(dp)  # Standard deviation of the distribution

print(mean, std_dev)

x = np.linspace(mean - 5 * std_dev, mean + 5 * std_dev, 1000)

y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# Create the plot with more bins for a smoother curve
plt.plot(x,y)

# Set labels and title
plt.xlabel('Accuracy', fontsize = 12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Average Total System Accuracy over 10 Experiments', fontsize=16)

# Show the plot
plt.show()
