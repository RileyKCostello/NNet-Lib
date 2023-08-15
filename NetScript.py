import numpy as np
import random

class layer:

    def __init__(self, numNeurons, numWeights, af = 0):
        self.af = af
        self.numNeurons = numNeurons
        self.numWeights = numWeights
        self.useSigmoid = sigmoid
        self.neurons = np.asmatrix(np.zeros(numNeurons))
        self.weights = np.asmatrix(np.zeros((numNeurons,numWeights)))
        self.biases = np.asmatrix(np.zeros((numWeights)))
        #Adjust matrices hold how much to change weights and biases after backpropagation
        self.weightAdjust = np.asmatrix(np.zeros((numNeurons, numWeights)))
        self.biasAdjust = np.asmatrix(np.zeros(numWeights))
        #derivs list used later in backProp
        self.derivs = []
        return

    #randomize every weight and bias as float between -1 and 1
    def rand(self):
        for ix, iy in np.ndindex(self.weights.shape):
            self.weights[ix,iy] = random.uniform(-1,1)
        for ix, iy in np.ndindex(self.biases.shape):
            self.biases[ix,iy] = random.uniform(-1,1)


    def next(self):
        output = self.neurons * self.weights
        output += self.biases
        self.setDerivs(output)
        output = self.actFunc(output)
        return output
        
    def setNeurons(self, values):
        self.neurons = np.asmatrix(values)

    #saves the derivatives of the squish function for easy access in backProp
    def setDerivs(self, a):
        self.derivs = self.afDeriv(np.asarray(a)[0])

    #Updates the weights and biases
    def adjust(self):
        self.weights -= self.weightAdjust
        self.biases -= self.biasAdjust
        #May be slow?
        self.weightAdjust[:] = 0
        self.biasAdjust[:] = 0
        return

    def actFunc(self, x):
        #tanh
        if self.af == 0:
            return np.tanh(x)
        #sigmoid
        if self.af == 1:
            for ix, iy in np.ndindex(x.shape):
                x[ix,iy] = sigmoid(x[ix,iy])
            return x
        #ReLu
        if self.af == 2:
            for ix, iy in np.ndindex(x.shape):
                x[ix,iy] = max(0, x[ix,iy])
            return x
        #Softmax
        if self.af == 3:
            a = np.asarray(x)[0]
            a = np.clip(a, -500, 500)
            return np.asmatrix(np.exp(a)/np.sum(np.exp(a)))

    def afDeriv(self, x):
        output = [0] * len(x)
        #tanh
        if self.af == 0:
            x = np.cosh(x)
            x = (1/x) ** 2
            return x
        #sigmoid
        if self.af == 1:
            for i,v in enumerate(x):
                output[i] = sigmoid(v) * (1-sigmoid(v))
            return np.asarray(output)
        #ReLu
        if self.af == 2:
            for i,v in enumerate(x):
                output[i] = 0
                if v > 0:
                    output[i] = 1
            return np.asarray(output)
        #Softmax
        if self.af == 3:
            output = [0] * len(x)
            x = np.asarray(x)
            x = np.clip(x, -500, 500)
            smax = np.exp(x)/np.sum(np.exp(x)).tolist()
            for i, v in enumerate(smax):
                j = 0
                while j < len(output):
                    if j == i:
                        output[j] += v * (1-v)
                    else:
                        output[j] += v * smax[j]
                    j += 1
            return np.asarray(output)
            

def sigmoid(x):
    if x < -500:
        return 0
    return 1/(1+np.exp(-x))

    
#Network initialized with an array or list
#Length determines the number of layers
#Elements determine number of neurons
'''af is the activation function used
0: tanh
1: sigmoid
2: ReLu
3: Softmax (last layer only)
'''
class network:
    def __init__(self, initList, batchSize, learnRate, af = 0, determ = True):
        if determ:
            random.seed(1426436)
        self.af = af
        self.cost = 0
        self.size = len(initList)
        self.layers = []
        self.outputs = []
        self.batchSize = batchSize
        self.learnRate = learnRate
        i = 0
        while i < self.size - 1:
            self.layers.append(layer(initList[i], initList[i+1], af))
            if random:
                self.layers[i].rand()
            i += 1

    def setInputs(self, inputs):
        self.layers[0].setNeurons(inputs)
        self.setOutputs()

    #Iterates through layers to calculate the output
    def setOutputs(self):
        self.outputs = []
        i = 0
        a = self.layers[0].next()
        while i < self.size - 2:
            self.layers[i+1].setNeurons(a)
            a = self.layers[i+1].next()
            i += 1
        for ix, iy in np.ndindex(a.shape):
            self.outputs.append(a[ix,iy])
            self.outputs[-1] = round(self.outputs[-1], 5)

    #Takes the expected outputs as a list and calculates the cost
    def getCost(self, answers):
        self.cost = 0
        i = 0
        while i < len(answers):
            a = self.outputs[i] - answers[i]
            a = a*a
            self.cost += a
            i += 1
        return self.cost / len(answers)

    '''nextLayer is a list of each nodes derivative in the next layer
    Bad naming convention as nextLayer corresponds to the
    previously calculated layer when back propagating '''
    def train(self, answers):
        nextLayer = []
        i = 0
        while i < len(answers):
            nextLayer.append(2*(self.outputs[i]-answers[i]))
            i += 1
        i = len(self.layers) - 1
        while i > -1:
            nextLayer = self.backProp(self.layers[i], nextLayer)
            i -= 1

    def setRelu(self):
        self.af = 2
        for a in layers:
            a.af = 2
        return

    def setSigmoid(self):
        self.af = 1
        for a in layers:
            a.af = 1
        return

    def setSoftmax(self):
        self.layers[-1].af = 3
    
    #Calculates the adjusts for weights and biases
    #Returns dervatives of the cost in terms of the neurons
    def backProp(self, a, nextLayer):
        #Find weight adjustment
        m1 = np.asmatrix(np.vstack(np.asarray(a.neurons)[0]))
        m2 = np.asmatrix(a.derivs * nextLayer)
        toAdd = m1 * m2 # Matrix multiply to get the derivatives
        toAdd = toAdd * (self.learnRate/ self.batchSize) #Array will be added to weightAdjust
        a.weightAdjust += toAdd
        #Find the derivative of neurons in this layer
        thisLayer = np.asarray(m2 * np.rot90(a.weights))[0]
        #Find Bias Adjustment
        for x, b in np.ndindex(a.biases.shape):
            a.biasAdjust[x, b] += self.learnRate * a.derivs[b] * nextLayer[b] * (1/self.batchSize)
        return thisLayer

    def updateLayers(self):
        for a in self.layers:
            a.adjust()
