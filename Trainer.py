import random
class Trainer:

    def __init__(self, mlp, trainingData, solutions):
        """
        Args:
            self (Trainer): self
            mlp (MLP): MLP object to train
            trainingData (Iterable conataining Real numbers): Object containing data
            solutions (Iterable containing labels): Contains the corrcet labels for each training object in the same order
        """
        self.mlp = mlp
        self.trainingData = trainingData
        self.solutions = solutions

    def train(self, numReps, step):
        for _ in range(numReps):

            # forward pass
            ypred = [self.mlp(x) for x in self.trainingData]

            # calculate loss
            loss = sum((yout - ygt)**2 for ygt, yout in zip(self.solutions, ypred))

            #backward pass
            #zero grad, resetting grads so they aren't accumulated from previous runs
            for p in self.mlp.parameters():
                p.grad = 0 

            loss.backward()

            # ONLY CHANGES VALUE IN PARAMETERS i.e WEIGHTS AND BIASES
            for p in self.mlp.parameters():
                p.data += -step * p.grad

    # def train(self, numReps, step, training_set):
    #     """ trains the MLP belonging to this Train object with the provided training set
    #         Args:
    #             numReps (int): number of times the model should be trained on the data from the training_set
    #             step (int): the rate at which the parameters should change relative to the gradient in back propagation
    #             training_set (list of tuple): a 2d array, each item contains a tuple where the first element represents the first training input \
    #                 and the second element in the tuple represents the solution.        
    #     """
    #     for _ in range(numReps):

    #         # forward pass
    #         ypred = [self.mlp(x[0]) for x in training_set]

    #         # calculate loss
    #         loss = sum((yout - ygt)**2 for ygt, yout in zip(training_set[], ypred))

    #         #backward pass
    #         #zero grad, resetting grads so they aren't accumulated from previous runs
    #         for p in self.mlp.parameters():
    #             p.grad = 0 

    #         loss.backward()

    #         # ONLY CHANGES VALUE IN PARAMETERS i.e WEIGHTS AND BIASES
    #         for p in self.mlp.parameters():
    #             p.data += -step * p.grad
        
    def batch_train(self, batch_size, reps, step):
        # zips solutions to their training data
        zipped_trainingset = zip(self.trainingData, self.solutions) # check what zip returns. should be numpy array
        # samples a random, non repeated selection
        sample_indices = random.sample(range(0, len(zipped_trainingset)-1), batch_size)

        # selects the row indices in the list 'sample_indices' and all columns within that index
        #i.e zipped_trainingset[sample_indices][all columns]
        batch = zipped_trainingset[sample_indices,:]
        # loop through indices list accessing correct items

        #self.train(reps, step)




    #batch training
    # choose a batch size, take this number of data points from the training data
    # run the network on all examples in this batch, measure the loss by averaging the individual losses
    # backpropagate the loss, update the weights at the end once. That is one batch
    # repeat this for the number of batches in the training data
    # cross entropy loss - better loss function for classification