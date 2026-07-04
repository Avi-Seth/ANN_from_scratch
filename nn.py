def synDiv(B, Poly): #Synthetic Division - Used with the remainder theorem to calculate output of polynomial function
    bd = 0
    qPoly = []
    for coeff in Poly:
        qPoly.append(coeff + bd)
        bd = B*(coeff + bd)
    return qPoly, qPoly[-1]

def fac(num): # Factorials
    result = 1
    for i in range (1, num+1):
        result *= i
    return result

def epower(acc): # Taylor series polynomial for e^x with variable Taylor Terms
    euler = []
    for i in range (0, acc):
        euler.append(1/fac(i))
    euler.reverse()
    return euler

standardEuler = epower(30) # Taylor series for e^x w/ 30 terms

def sig(x): # Sigmoid Function
    ep = synDiv(x, standardEuler)[1]
    return (1/(1 + (1/ep)))


class CSN_NeuralNetwork():
    def __init__(self, name, layers, outputs): # Neural network class defined using an array describing layers, neurons in each layer, and output labels.
        self.name = name
        self.layers = layers
        self.outputs = outputs
        if len(outputs) != layers[-1]:
            print(f"NNError: neural network '{name}' is not instantiable due to size conflict between output layer and output labels")
            raise SystemExit
        self.weights = [[[0.5 for i in range(layers[n])] for j in range(layers[n+1])] for n in range(len(layers) - 1)] # Starting in the middle
        self.biases = [[0 for j in range(layers[n+1])] for n in range(len(layers) - 1)] # Bias vectors

    def __str__(self): # Our neural network is defined by the weight matrices and bias vectors between each layer, so that is how we want to display it
        display = ''
        count = 0
        display += 'WEIGHTS:\n'
        for weightmatrix in self.weights:
            display += (f'|========LAYER {count}-{count+1}========\n')
            for row in weightmatrix:
                display += (f'|{row}\n')
            count += 1

        count = 0
        display += 'BIASES:\n'
        for bias in self.biases:
            display += (f'|========LAYER {count}-{count+1}========\n')
            display += (f'|{bias}\n')
            count += 1

        return display

    def test(self, inputarray):
        if len(inputarray) != self.layers[0]:
            print(f"NNError: cannot test neural network '{self.name}' due to size conflict between input stimulus and input layer")
            raise SystemExit

        activation = inputarray # Initial activations are in input layer

        for layer in range(len(self.layers)-1):
            newactivation = []
            weightmatrix = self.weights[layer] # Appropriate weight matrix for our current layer
            count = 0
            for row in weightmatrix:
                sum = 0
                for a in range(len(activation)):
                    sum += row[a]*activation[a]
                newactivation.append(sum + self.biases[layer][count]) # Adding the neuron activation as an element to our next activation layer
                count += 1
            activation = newactivation

        final = [] # Converting the output layer activations to a vector of probabilities
        for element in activation:
            final.append(sig(element))

        max = final[0] # Finding the list of all values in the
        conflict = False
        for value in final:
            if value > max:
                max = value
            elif value == max:
                conflict = True

        conlist = []
        iter = len(conlist) + 1
        for i in range(len(final)):
            if final[i] == max:
                conlist.append(i)

            conflict = len(conlist) > 1      # only a real conflict if 2+ indices share the max

            if conflict:
                names = [self.outputs[i] for i in conlist]
                statement = "Output conflicting between " + ", ".join(names[:-1]) + ", and " + names[-1]
            else:
                statement = self.outputs[conlist[0]]

            return final, statement

    def train(self, batch, labels, step): # Training our model using batches of data
        sample = 0
        avgCostGradientWeights = [[[0 for i in range(self.layers[n])] for j in range(self.layers[n+1])] for n in range(len(self.layers) - 1)]
        avgCostGradientBiases = [[0 for i in range(len(self.biases[j]))] for j in range(len(self.biases))]

        for inputarray in batch:
            if len(inputarray) != self.layers[0]:
                print(f"NNError: cannot test neural network '{self.name}' due to size conflict between input stimulus and input layer")
                raise SystemExit

            activation = inputarray
            activationSet = [inputarray]

            for layer in range(len(self.layers)-1):
                newactivation = []
                weightmatrix = self.weights[layer]
                count = 0
                for row in weightmatrix:
                    sum = 0
                    for a in range(len(activation)):
                        sum += row[a]*activation[a]
                    newactivation.append(sum + self.biases[layer][count])
                    count += 1
                activation = newactivation
                activationSet.append(newactivation)

            final = []
            for element in activation:
                final.append(sig(element))

            if labels[sample] not in self.outputs:
                print(f"NNError: label '{labels[sample]}' (Sample {sample}) does not exist in the possible outputs of '{self.name}'")
                raise SystemExit
            #====================================================================

            rule = [(1 if self.outputs[i] == labels[sample] else 0) for i in range(len(self.outputs))] # Matching the sample with output

            deltas = None  # Holds the delta vector of the layer just processed (one layer closer to output)

            for n in range(-1, -len(self.layers), -1):
                newDeltas = []
                for m in range(len(activationSet[n])):
                    if n == -1:
                        # Output layer is squashed -> needs the sigmoid derivative
                        a = final[m]
                        dC_da = 2 * (a - rule[m])
                        da_dz = a * (1 - a)
                        delta = dC_da * da_dz
                    else:
                        # Hidden layers are linear -> derivative is 1, just propagate the next layer's deltas back
                        delta = 0
                        for j in range(len(deltas)):
                            delta += deltas[j] * self.weights[n + 1][j][m]
                    newDeltas.append(delta)

                # Accumulate gradients for the weights/biases feeding into this layer
                for j in range(len(newDeltas)):
                    for i in range(len(activationSet[n - 1])):
                        avgCostGradientWeights[n][j][i] += newDeltas[j] * activationSet[n - 1][i]
                    avgCostGradientBiases[n][j] += newDeltas[j]

                deltas = newDeltas

            sample += 1

        # Take avg over batch
        for n in range(len(avgCostGradientWeights)):
            for j in range(len(avgCostGradientWeights[n])):
                for i in range(len(avgCostGradientWeights[n][j])):
                    self.weights[n][j][i] -= step * (avgCostGradientWeights[n][j][i] / len(batch))

        for n in range(len(avgCostGradientBiases)):
            for j in range(len(avgCostGradientBiases[n])):
                self.biases[n][j] -= step * (avgCostGradientBiases[n][j] / len(batch))


base = CSN_NeuralNetwork("baseline", [5, 4, 3, 2], ["A", "B"])
a = base.train([[1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 1],
                [1, 1, 1, 0, 0],
                [0, 1, 1, 0, 0]],
                ["A", "B", "A", "A", "A", "A", "A", "B", "B"], 0.1)
t = base.test([0, 0, 1, 1, 1])[1]
print(base)