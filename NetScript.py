import numpy as np
import random


class layer:

    def __init__(self, numNeurons, numWeights):
        self.neurons = np.asmatrix(np.zeros(numNeurons))
        self.weights = np.asmatrix(np.zeros((numNeurons,numWeights)))
        self.biases = np.asmatrix(np.zeros((numWeights)))
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
        output = np.tanh(output)
        return output
        
    def setNeurons(self, values):
        self.neurons = values
        
#Network initialized with an array or list
#Length determines the number of layers
#Elements determine number of neurons
class network:
    def __init__(self, initList, random = True):
        self.cost = 0
        self.size = len(initList)
        self.layers = []
        self.outputs = []
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
        return self.cost
        
