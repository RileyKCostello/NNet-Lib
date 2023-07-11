import numpy as np
import random

class layer:

    def __init__(self, numNeurons, numWeights):
        self.numNeurons = numNeurons
        self.numWeights = numWeights
        self.neurons = np.asmatrix(np.zeros(numNeurons))
        self.weights = np.asmatrix(np.zeros((numNeurons,numWeights)))
        self.biases = np.asmatrix(np.zeros((numWeights)))
        #Adjust matrices hold how much to change weights and biases after backpropagation
        self.weightAdjust = np.asmatrix(np.zeros((numNeurons, numWeights)))
        self.biasAdjust = np.asmatrix(np.zeros(numWeights))
        #derivs list used later in backProp
        self.derivs = []
        return

    #randomize every weight and bias as float between -2 and 2
    def rand(self):
        for ix, iy in np.ndindex(self.weights.shape):
            self.weights[ix,iy] = random.uniform(-2,2)
        for ix, iy in np.ndindex(self.biases.shape):
            self.biases[ix,iy] = random.uniform(-2,2)

    def next(self):
        output = self.neurons * self.weights
        for ix, iy in np.ndindex(output.shape):
            output[ix,iy] += self.biases[ix,iy]
        self.setDerivs(output)
        output = np.tanh(output)
        return output
        
    def setNeurons(self, values):
        self.neurons = np.asmatrix(values)

    #saves the derivatives of the tanh function for easy access in backProp
    def setDerivs(self, a):
        self.derivs = np.asarray(a)[0]
        self.derivs = np.cosh(self.derivs)**2
        self.derivs = 1/self.derivs

    #Updates the weights and biases by adding with weight.adjust and bias.adjust
    def adjust(self):
        for ix, iy in np.ndindex(self.weights.shape):
            self.weights[ix, iy] -= self.weightAdjust[ix, iy]
            self.weightAdjust[ix,iy] = 0
        for ix, iy in np.ndindex(self.biases.shape):
            self.biases[ix,iy] -= self.biasAdjust[ix,iy]
            self.biasAdjust[ix,iy] = 0
        return


#Network initialized with an array or list
#Length determines the number of layers
#Elements determine number of neurons
class network:
    def __init__(self, initList, batchSize, learnRate, random = True):
        self.cost = 0
        self.size = len(initList)
        self.layers = []
        self.outputs = []
        self.batchSize = batchSize
        self.learnRate = learnRate
        i = 0
        while i < self.size - 1:
            self.layers.append(layer(initList[i], initList[i+1]))
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
    
    #Calculates the adjusts for weights and biases
    #Returns dervatives of the cost in terms of the neurons
    def backProp(self, a, nextLayer):
        #Find weight adjustment
        m1 = np.asmatrix(np.vstack(np.asarray(a.neurons)[0]))
        m2 = np.asmatrix(a.derivs * nextLayer)
        toAdd = m1 * m2 # Matrix multiply to get the derivatives
        toAdd = toAdd * (self.learnRate/ self.batchSize) #Create the array that will be added to weightAdjust
        a.weightAdjust += toAdd
        #Find the derivative of neurons in this layer
        thisLayer = np.asarray(m2 * np.rot90(a.weights))[0]
        #Find Bias Adjustment
        for x, b in np.ndindex(a.biases.shape):
            #print(nextLayer[b])
            a.biasAdjust[x, b] += self.learnRate * a.derivs[b] * nextLayer[b] * (1/self.batchSize)
        return thisLayer

    def updateLayers(self):
        for a in self.layers:
            a.adjust()
