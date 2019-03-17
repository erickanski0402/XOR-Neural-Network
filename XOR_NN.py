import numpy as np
import neural_network as network
import matplotlib.pyplot as plt

nn = network.neural_network(2,4,1,0.5)
print("Initial input_hidden weights:\n", nn.weights_ih)
print("Initial hidden_output weights:\n", nn.weights_ho)

inputs = [[0,0],[0,1],[1,0],[1,1]]
targets = [[0],[1],[1],[0]]
# targets = [[0,0],[1,0],[0,1],[0,0]]

i = 0
for example in inputs:
    print("Guess: ", nn.query(example), "Actual: ", targets[i])
    i += 1

print("")

for _ in range(1000):
    for i in range(len(inputs)):
        nn.train(inputs[i],targets[i])

print("Post input_hidden weights:\n", nn.weights_ih)
print("Post hidden_output weights:\n", nn.weights_ho)

i = 0
for example in inputs:
    print("Guess: ", nn.query(example), "Actual: ", targets[i])
    i += 1

nn.visualize()
