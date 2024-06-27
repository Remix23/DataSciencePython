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

comb_train = loadCombined(training=True, normalise=True, convert_labels = True)

test1.performRunTraining(comb_train, opt, opt, batch_size=BATCH_SIZE, examine_progress=True, num_of_epochs = 30)

comb_test = loadCombined(training = False, normalise = True)

print(test1.performRunTest(comb_test))