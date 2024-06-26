### loaders
from data.DataSets.MNIST.loader import loadCombined, showGreyIMG, convertLabels

### networks
from ANNS.simple import SimpleANN
from ANNS.utils import *

BATCH_SIZE = 10

### set up network 
layers = [
    784, 
    30,
    10
]


act = Sigmoid()
init = Normal()
out_func = SoftMax()
loss_func = MSE()

opt = SGD(3, BATCH_SIZE)

### create testing network:
test1 = SimpleANN(layers, 
                  activation_func = act,
                  initializer = init, 
                  ouput_func = out_func, 
                  loss_func = loss_func)


# m1 = np.arange(1, 9).reshape((4, 2))
# m2 = np.arange(1, 9).reshape((2,4))

# test1.weight_matrices = [m1, m2]

# print(test1.propagateForward(np.ones(2)))
# print(test1.propagateBackwards(np.array([1, 0])))

imgs, labels = loadCombined(training=True, normalise=True)

labels = convertLabels(labels)

test1.performRunTraining(imgs, labels, opt, opt, batch_size=BATCH_SIZE)

imgs, labels = loadCombined(training = False, normalise = True, n_max = 1000)

print(test1.performRunTest(imgs, labels))