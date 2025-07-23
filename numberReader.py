import gzip
import idx2numpy as idx
import numpy as np

import MLP
import Trainer

localFP = "vscodeMicroGrad\\nonmatlab\\digits\\"

filenames = [
    localFP + "emnist-digits-train-images-idx3-ubyte.gz",
    localFP + "emnist-digits-test-images-idx3-ubyte.gz",
    localFP + "emnist-digits-train-labels-idx1-ubyte.gz",
    localFP + "emnist-digits-test-labels-idx1-ubyte.gz"
]

#unzips the binary files and stores them
unzipTrainingImg = gzip.open(filenames[0])
unzipTrainingLb = gzip.open(filenames[2])
unzipTestImg = gzip.open(filenames[1])
unzipTestLb = gzip.open(filenames[3])

# converts from the binary .idx files into numpy arrays
# More specifically, into an array with 240,000 slots, each containing another 28x28 2d array representing the images.
trainingImages = idx.convert_from_file(unzipTrainingImg)
trainingLbs = idx.convert_from_file(unzipTrainingLb)

#print(trainingLbs.shape)

# makes the array 1 dimensional
# the '-1' means that this dimension is unspecified. We know that there are 240,000 images so we know that this dimension is 240,000
array1d = trainingImages.reshape(240000, 28*28)

xs = array1d/255

ys = trainingLbs

mlp = MLP.MLP(784, [10, 10, 1])

trainer = Trainer.Trainer(mlp, xs, ys)

#trainer.train(1, 0.5)


#todo:
#   implement batch training as an option in the train method
#   save parameters in csv
#   read in parameters in constructor?

# LOSS FUNCTION:
#   let the user pass in a loss function to the train method
#   Cross-entropy loss function instead of MSE loss

#