# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:02:49 2025

@author: Max Robinson
"""
import math
from graphviz import Digraph
#%matplotlib inline

class Value:
    def __init__(self, data, _children = (), _op = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda : None
        self._op = _op
    
    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int or float powers"
        out = Value(self.data**other, (self, ), f"**{other}")
        
        def _backward():
            self.grad += other * self.data ** (other-1) * out.grad
        out._backward = _backward
        
        return out
    
    def __truediv__(self, other):
        return self * other**-1          
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        n = self.data
        
        tanh = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        
        out = Value(tanh, (self, ), 'tanh')
        
        def _backward():
            #out.grad will always start at 0
            self.grad += 4 / (math.exp(2*n) + math.exp(-2*n) + 2) * out.grad
                     
        out._backward = _backward
        
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                topo.append(v)
                for child in v._prev:
                    build_topo(child)
        build_topo(self)
        self.grad = 1.0
        for node in topo:
            node._backward()

# builds the graph's nodes and edges set
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format = 'svg', graph_attr = {'rankdir': 'LR'})

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{data %.4f}" % (n.data, ), shape = "record")
        # '' evaluates to false
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid+n._op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

a = Value(2)
b = Value(-2)
c = Value(-1)

e = a*b + c
f = Value(2)
#print(a/b)
L = e + f # i.e. loss function (how far off the output is from being correct)
# manual back progation
L.grad = 1.0 # dL/dL = 1
f.grad = 1.0 # dL/df = 1
e.grad = 1.0 # dL/de = 1

a.grad = e.grad*b.data # dL/da = dL/de * de/da
b.grad = e.grad*a.data # dL/db = dL/de * de/db
c.grad = 1.0           # dL/dc = dL/de * de/dc


# building a neuron

# inputs x1, x2
x1 = Value(2.0)
w1 = Value(-3.0)
x2 = Value(-0.0)
w2 = Value(1.0)
b = Value(6.88137)

# sum of inputs with their weights and bias added. i.e. Modelling the neuron.

x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b # value exiting the neuron
o = n.tanh() # squashing n between -1 and 1
#or
e = (2*n).exp()
o = (e - 1)/ (e + 1)

#print(o)
# back propagating the neuron
o.backward()
#print(w1.grad)

# sech^2(n) = 4/(e^2n + e^-2n + 2)
# n.grad = 4 / (math.exp(2*n.data) + math.exp(-2*n.data) + 2)
# b.grad = n.grad * 1.0 # do/dn * dn/db
# x1w1x2w2.grad = n.grad * 1.0
# x1w1.grad = x1w1x2w2.grad * 1.0
# x2w2.grad = x1w1x2w2.grad * 1.0

# x1.grad = x1w1.grad * w1.data # x1 * w1 = x1w1
# w1.grad = x1w1.grad * x1.data
# x2.grad = x2w2.grad * w2.data
# w2.grad = x2w2.grad * x2.data

#draw_dot(o)

#------------------------------------------------------------------------------
# Using pytorch API
# import torch

# x1 = torch.Tensor([2.0]).double() #from float32 to float64
# x1.requires_grad = True
# w1 = torch.Tensor([-3.0]).double();      w1.requires_grad = True
# x2 = torch.Tensor([0.0]).double();       x2.requires_grad = True
# w2 = torch.Tensor([1.0]).double();       w2.requires_grad = True
# b = torch.Tensor([6.88137358]).double(); b.requires_grad = True

# n = x1*w1 + x2*w2
# o = torch.tanh(n)

#print(o.data.item())
#------------------------------------------------------------------------------
#neural net
import random
class Neuron:
    def __init__(self, nin):
        # for each input (nin) create a random weighting save it in this array
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1)) # random bias between -1 and 1
    
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
    
x = [2.0, 3.0]
n = Neuron(2)
#print(n(x)) # this will call __call__

class Layer:
    def __init__(self, nin, nout): # nin: number of inputs per neuron inthis layer (dimensionality of neurons), nout: number of outputs to this layer (i.e. # of neurons)
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x): # x: data to call each neuron with. (i.e. all the data)
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        #params = []
        #for neuron in self.neurons:
        #    ps = neuron.parameters()
        #    params.extend(ps)
        #return params
    
class MLP:
    def __init__(self, nin, nouts): # nin: # of input, nouts: array of # outputs for every layer
        sz = [nin] + nouts # make an array of [nin, nouts]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) #calculates the output of the prev layer and passes it as input of the next layer
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# 2 inputs, 3 outputs/ 3 neurons
l = Layer(2, 3)
#print(l(x))
# multi-layer perceptron with 3 layers, sized 4, 4 and 1 neuron and 3 inputs
mlp = MLP(3, [4, 4, 1])

# -----example data-----------------------------------------------------------------------------
# inputs
xs = [[2.0, 3.0, -1.0], # one of these represents the three inputs to a single run of the network
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]

# desired outputs
ys = [1.0, -1.0, -1.0, 1.0]

# use the mean square error loss. 
# I.e. find the distance between the target and the predicted value
# gradient descent
for k in range(30): # 10 training steps
    # forward pass
    ypred = [mlp(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    #backward pass
    #zero grad, resetting grads so they aren't accumulated from previous runs
    for p in mlp.parameters():
        p.grad = 0 

    loss.backward()

    #update
    for p in mlp.parameters():
        #print(f"before={p.data}")
        p.data += -0.1 * p.grad
        #print(f"after={p.data}")

    #print(k, loss)

#print(ypred)

print(mlp(xs[0]))
print(mlp(xs[1]))
print(mlp(xs[2]))
print(mlp(xs[3]))
print(f"loss={loss}")