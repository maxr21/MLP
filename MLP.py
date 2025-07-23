import Value
import random

class Neuron:
    def __init__(self, nin):
        # for each input (nin) create a random weighting save it in this array
        self.w = [Value.Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value.Value(random.uniform(-1,1)) # random bias between -1 and 1
    
    def __call__ (self, x):
        # w * x + b
        #sum
        #for wi, xi in zip(self.w, x):
        #    sum += xi*wi
        activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, nin, nout): # nin: number of inputs per neuron inthis layer (dimensionality of neurons), nout: number of outputs to this layer (i.e. # of neurons)
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x): # x: data to call each neuron with. (i.e. all the data)
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts): # nin: # of input, nouts: array of # outputs for every layer
        sz = [nin] + nouts # make an array of [nin, nouts]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self._nin = nin
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) #calculates the output of the prev layer and passes it as input of the next layer
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
