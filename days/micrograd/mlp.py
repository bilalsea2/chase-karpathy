import random
from grad import Value

class Neuron:

    def __init__(self, nin):
        self.w = [(Value(random.uniform(-1, 1))) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b).tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    
    def __init__(self, nin, nout): # nout -> number of neurons in the layer
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            neuron_params = neuron.parameters()
            params.extend(neuron_params)
        return params

class MLP:

    def __init__(self, nin, nouts): 
        # nouts -> list of nouts, nout_i is number of neurons for layer_i
        sz = [nin] + nouts 
        # sz = [nin, nout_0, nout_1, ...] -> layer_0's outputs are layer_1's inputs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            layer_params = layer.parameters()
            params.extend(layer_params)
        return params
