import math
import numpy as np
import scipy.special
from graphics import *

class neural_network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, alpha):
        # Number of nodes per layer as given
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # Learning rate
        self.lr = alpha

        # Weights randomly generated between -1 and 1 based on the number of nodes given
        self.weights_ih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.weights_ho = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # Assigning sigmoid function (already defined in scipy.special) as a lambda function
        self.activation_function = lambda x: scipy.special.expit(x)
        pass


    # Backpropagation
    def train(self, inputs_list, targets_list):
        # Convert the list of inputs and targets passed into numpy vectors
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T


        # Calculate the weighted sum going into the set of nodes in the hidden layer
        hidden_inputs = np.dot(self.weights_ih, inputs)
        # Activate each weighted sum
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate the weighted sum going into the set of nodes in the output layer
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        # Activate each weighted sum
        final_outputs = self.activation_function(final_inputs)


        # Overall error of the output nodes
        output_errors = targets - final_outputs
        # Error of the hidden nodes
        hidden_errors = np.dot(self.weights_ho.T, output_errors)


        # Update the weights for the links between hidden and output layers
        self.weights_ho += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
        # gradient of w_jk = alpha * Error_output * d_sigmoid(final_outputs) * transpose(final_outputs)

        # Update the weights for the links between input and hidden layers
        self.weights_ih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
        # gradient of w_ij = alpha * Error_hidden * d_sigmoid(hidden_outputs) * transpose(inputs)
        pass

    def query(self, inputs):
        # Calculate the weighted sum going into the set of nodes in the hidden layer
        hidden_inputs = np.dot(self.weights_ih, inputs)
        # Activate each weighted sum
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate the weighted sum going into the set of nodes in the output layer
        final_inputs = np.dot(self.weights_ho, hidden_outputs)
        # Activate each weighted sum
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    def visualize(self):
        # Create window
        win = GraphWin('Neural Network',400,300)

        radius = 15

        inputs = []
        hiddens = []
        outputs = []

        for _ in range(self.onodes):
            outputs.append(Circle(Point(300, 50 * _ + 50), radius))
            outputs[_].draw(win)

        for _ in range(self.hnodes):
            hiddens.append(Circle(Point(200, 50 * _ + 50), radius))
            hiddens[_].draw(win)

            for i in range(self.onodes):
                p1 = self.getEdgePoint(hiddens[_],outputs[i],radius)
                p2 = self.getEdgePoint(outputs[i],hiddens[_],radius)
                lines_who = Line(p1, p2)
                lines_who.draw(win)

        for _ in range(self.inodes):
            inputs.append(Circle(Point(100,100 * _ + 50), radius))
            inputs[_].draw(win)

            for j in range(self.hnodes):
                p1 = self.getEdgePoint(inputs[_],hiddens[j],radius)
                p2 = self.getEdgePoint(hiddens[j],inputs[_],radius)
                lines_wih = Line(p1,p2)
                lines_wih.draw(win)

        # Wait for the user to make an input before closing the window
        win.getMouse()
        win.close()


    def getEdgePoint(self, circleA, circleB, r):
        cartesianDistance = math.sqrt(math.pow((circleB.getCenter().getX() - circleA.getCenter().getX()),2) + math.pow((circleB.getCenter().getY() - circleA.getCenter().getY()),2))
        C_x = circleA.getCenter().getX() + r * (circleB.getCenter().getX() - circleA.getCenter().getX()) / cartesianDistance
        C_y = circleA.getCenter().getY() + r * (circleB.getCenter().getY() - circleA.getCenter().getY()) / cartesianDistance
        return Point(C_x,C_y)
