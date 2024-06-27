import numpy as np

from .utils import *

from random import sample

class SimpleANN:

    def __init__(self, layers : list[int], activation_func : ActivationFunc, initializer : Initializer = Uniform(), bias_initializer : Initializer = Uniform(0, 0), ouput_func : TransformFunc = SoftMax(), loss_func : LossFunc = MSE()) -> None:

        self.weight_matrices : list[np.ndarray] = [] # arr of n - 1 matrices each of size n_i x n_i+1 initialized with random weights
        self.bias_matrices : list[np.ndarray] = []
        self.cost : np.ndarray = np.zeros(layers[-1])

        ### gen neurons - values to be passed futher
        self.layers_info = layers
        self.layers : list[np.ndarray] = []

        ### value for backpropagation
        self.zs : list[np.ndarray] = []

        self.num_of_layers = len(layers)

        self.activation_func = activation_func
        self.initializer = initializer
        self.bias_initializer = bias_initializer
        self.out_put_func = ouput_func
        self.loss_func = loss_func

        self.genLayers()
        self.genWeights()
        self.genBiases()

    def genLayers (self) -> None:
        self.layers = [np.zeros(num) for num in self.layers_info]
        self.zs = [np.zeros(num) for num in self.layers_info]

    def genWeights (self) -> None:
        self.weight_matrices.clear()
        for i in range(self.num_of_layers - 1):
            # m = np.random.uniform(0, 1, size = (len(self.layers[i + 1]), len(self.layers[i])))
            shape = len(self.layers[i + 1]), len(self.layers[i])
            m = self.initializer.getNext(shape = shape)
            self.weight_matrices.append(m)

    def genBiases (self) -> None:
        self.bias_matrices.clear()
        for i in range(1, self.num_of_layers):
            shape = len(self.layers[i])
            bs = self.bias_initializer.getNext(shape = shape)
            bs = bs.flatten()
            # bs = np.random.uniform(0, 1, size = len(self.layers[i]))

            self.bias_matrices.append(bs)

    def propagateForward (self, data_in : np.ndarray) -> np.ndarray:
        if len(data_in) != len(self.layers[0]): raise(Exception(f"Dimension of the input data not correct: {len(data_in)}, should be {len(self.layers[0])}"))

        if data_in.shape != self.layers[0].shape:
            ### check if reshape possible
            data_in = np.reshape(data_in, self.layers[0].shape)

        self.layers[0] = data_in

        layer = 0

        for biases, weights in zip(self.bias_matrices, self.weight_matrices):

            out = np.matmul(weights, self.layers[layer])

            out += biases

            self.zs[layer + 1] = out

            self.layers[layer + 1] = np.array([self.activation_func.getVal(x) for x in out])

            layer += 1
        
        ### normalize with softmax
        self.layers[-1] = self.out_put_func.tranform(self.layers[-1])

        return self.layers[self.num_of_layers - 1]
    
    def propagateBackwards2 (self, desired_out : np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:

        ## create change tables
        bias_changes = [np.zeros(x) for x in self.layers_info[1:]]
        weight_changes = [np.zeros(s.shape) for s in self.weight_matrices]

        self.layers[-1] = -1 * self.loss_func.computeLossDerivative (self.layers[-1], desired_out)

        for i_layer in range(self.num_of_layers - 2, -1, -1):

            ### compute list of derrivatives w.r.t. activation func
            layer_dadz = self.activation_func.getDerrivative(self.zs[i_layer + 1])

            bias_changes[i_layer] = layer_dadz * self.layers[i_layer + 1]

            ### compute updates for weights and neurons
            dzdw = self.layers[i_layer]
            dcda = self.layers[i_layer + 1]

            weight_update = np.dot((dcda * layer_dadz).reshape(self.layers_info[i_layer + 1], 1), dzdw.reshape(1, self.layers_info[i_layer])) # self.info_layers[i_layer + 1] x self.info_layers[i_layer]
            weight_changes[i_layer] = weight_update

            neurons_update = np.sum(weight_update, axis = 0)
            self.layers[i_layer] = neurons_update

        return bias_changes, weight_changes

    def propagateBackwards (self, desired_out : np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if desired_out.shape != self.layers[-1].shape:
            raise Exception(f"Incorrect dimension of the `out` array: should be {len(self.layers[-1])} but was {len(desired_out)}" )

        ## create change tables
        bias_changes = [np.zeros(x) for x in self.layers_info[1:]]
        weight_changes = [np.zeros(s.shape) for s in self.weight_matrices]

        self.layers[-1] = -1 * self.loss_func.computeLossDerivative (self.layers[-1], desired_out)

        for i_layer in range(self.num_of_layers - 1, 0, -1):

            for i_neuron in range(len(self.layers[i_layer])):

                ### update neuron value for futher backpropagation steps and weights
                if i_layer != self.num_of_layers - 1:
                    s = 0
                    for next_layer in range(len(self.layers[i_layer + 1])): # next_layer indexes of neurons in the layer + 1
                        ### update weight [i_neuron] -> [next_layer]
                        ### i - index of curr layer | j - index of forward layer
                        dz_jdw_ji = self.layers[i_layer][i_neuron]
                        da_jdz_j = self.activation_func.getDerrivative(self.zs[i_layer + 1][next_layer])
                        dc_0da_j = self.layers[i_layer + 1][next_layer]

                        s += self.weight_matrices[i_layer][next_layer][i_neuron] * da_jdz_j * dc_0da_j

                        weight_change = dz_jdw_ji * da_jdz_j * dc_0da_j
                        #self.weight_matrices[i_layer][next_layer][i_neuron] += weight_change
                        weight_changes[i_layer][next_layer][i_neuron] = weight_change

                    self.layers[i_layer][i_neuron] = s

                if i_layer >= 1: 
                ### compute the common backprop parts
                    dadz = self.activation_func.getDerrivative(self.zs[i_layer][i_neuron])
                    dcda = self.layers[i_layer][i_neuron]

                    bias_changes[i_layer - 1][i_neuron] = dadz * dcda

        return (bias_changes, weight_changes)
    
    def proccessBatch (self, batch, bias_optimizer : Optimizer, weights_optimizer : Optimizer):
        biases_change_total = [np.zeros(x) for x in self.layers_info[1:]]
        weights_changes_total = [np.zeros(s.shape) for s in self.weight_matrices]

        for img, label in batch:
            # self.propagateForward(img)
            # a1, b1 = self.propagateBackwards(label)

            self.propagateForward(img)
            a2, b2 = self.propagateBackwards2(label)

            for i_layer in range(self.num_of_layers - 1):
                biases_change_total[i_layer] += a2[i_layer]
                weights_changes_total[i_layer] += b2[i_layer]

        for i_layer in range(self.num_of_layers - 1):
            self.bias_matrices[i_layer] += bias_optimizer.updateWeight(biases_change_total[i_layer])
            self.weight_matrices[i_layer] += weights_optimizer.updateWeight(weights_changes_total[i_layer])
        
        return biases_change_total, weights_changes_total

    def performRunTraining (self, data : list[tuple[np.ndarray, np.ndarray]], 
                            bias_optimizer : Optimizer, weights_optimizer : Optimizer, 
                            batch_size = 1, num_of_epochs = 1, examine_progress = False):
        n = len(data)

        if n % batch_size != 0: return ValueError("Wrong batch size, it is not possible to divide the original data")

        n_of_batches = n // batch_size

        if n == 0: raise ValueError("0 zero training examples")

        if n != len(data):
            raise ValueError("Incorrect number of validations")
        
        for i_epoch in range(num_of_epochs):
            
            np.random.shuffle(data)

            batches = [data[k : k + batch_size] for k in range(0, n, batch_size)]

            for i_batch in range(n_of_batches):

                self.proccessBatch(batches[i_batch], bias_optimizer , weights_optimizer)

            if examine_progress:
                ## run tests on randomly selected batch
                samples = sample(data, 1000)

                ech_score = self.performRunTest(samples)
                print(f"Epoch: {i_epoch} - score: {ech_score}")

    def performRunTest (self, data):
        corr = 0
        for img, lab in data:

            if type(lab) != np.uint8:
                lab = np.argmax(lab)

            model_prediction = np.argmax(self.propagateForward(img))

            if model_prediction == lab:
                corr += 1

        return corr / len(data)

    def printState (self, n) -> None:
        print(self.layers[n])

    def printWeights (self, n) -> None:
        print(self.weight_matrices[n])

    def __str__(self) -> str:
        n_weights = sum([np.prod(x.shape) for x in self.weight_matrices])
        n_neuros = sum([len(x) for x in self.layers])
        return f"""Number of layers: {self.num_of_layers}
            Sizes of layers: {",".join([str(len(x)) for x in self.layers])}
            Num of params:
            - Weights: {n_weights}
            - Biases: {n_neuros}
            Total: {n_weights + n_neuros}"""

if __name__ == "__main__":

    size_of_layers = [3, 5, 2]

    t = SimpleANN(size_of_layers, 
                  activation_func = ReLu(), 
                  initializer = Uniform(), 
                  ouput_func = SoftMax(), 
                  loss_func = MSE())

    ### img test