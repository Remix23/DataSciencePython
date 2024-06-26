### loaders
from data.DataSets.MNIST.loader import loadCombined, showGreyIMG, convertLabels

### networks
from ANNS.simple import SimpleANN
from ANNS.utils import *


### performs simple test:
test1 = SimpleANN([784, 16, 10], ReLu(), Uniform(), ouput_func = SoftMax(), loss_func = MSE())

imgs, labels = loadCombined(training = True, normalise = True)

labels = convertLabels(labels)

for img, label in zip(imgs, labels):
    test1.propagateForward(img)
    test1.propagateBackwards(label)
