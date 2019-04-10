import tensorflow as tf
import random
import numpy as np

batchSize = 100

numClasses = 2

hiddenUnits = 50

numTrainingIters = 5000

TABLE_SIZE = 2500
#
# x_train = []
#
# y_train = []
#
#
# x_test = []
#
# y_test = []


def generateBatchData(x, y):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.random_integers(0, len(x) - 1, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x_train = np.stack(np.array((x[i])) for i in myInts.flat)

    #
    # and stack all of the labels into a vector of labels
    y_train = np.stack(np.array((y[i])) for i in myInts.flat)
    #
    # return the pair
    return (x_train, y_train)

def h(x):
    return ((37 * x + 47) % 2038072819) % TABLE_SIZE

table = [0 for i in range(TABLE_SIZE)]

def insert(x):
    idx = h(x)
    table[idx] = 1

def test(x):
    idx = h(x)
    if table[idx] == 1:
        return True
    else:
        return False

def load_table(percentage):
    for i in range(int(TABLE_SIZE * percentage)):
        item = random.randrange(0, TABLE_SIZE * 5)
        insert(item)

load_table(0.65)


def generate_random_data():
    x_set = set([])
    x = []
    y = []
    hits = 0
    miss = 0
    for i in range(TABLE_SIZE * 2):
        datum = random.randint(0, TABLE_SIZE * 3)
        if datum not in x_set:
            x.append(datum)
            if test(datum):
                hits += 1
                y.append(1)
            else:
                miss += 1
                y.append(0)
    length = len(x)
    x_train = np.asarray(x[: int(length * 0.8)])
    y_train = np.asarray(y[: int(length * 0.8)])
    x_test = np.asarray(x[int(length * 0.8):])
    y_test = np.asarray(y[int(length * 0.8):])
    print "hits = ", hits
    print "miss = ", miss
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = generate_random_data()


print x_train
print y_train
print x_test
print y_test

## Tuning the training data to be balanced

# a sequence of numbers representing the items to test the cache
inputX = tf.placeholder(tf.float32, [batchSize, 1])

# inputy is a sequence of zeroes and ones
inputY = tf.placeholder(tf.int32, [batchSize])

# the weight matrix that maps the inputs to hiddden layer
W = tf.Variable(np.random.normal(0, 0.05, (1, hiddenUnits)), dtype=tf.float32)
# biaes for the hidden values
b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

# weights and bias for the final classification
W2 = tf.Variable(np.random.normal (0, 0.05, (hiddenUnits, numClasses)), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,numClasses)), dtype=tf.float32)

hiddenLayer = tf.tanh(tf.matmul(inputX, W) + b)
outputs = tf.matmul(hiddenLayer, W2) + b2

predictions = tf.nn.softmax(outputs)

# compute the loss
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputY)
totalLoss = tf.reduce_mean(losses)

# use gradient descent to train
# trainingAlg = tf.train.GradientDescentOptimizer(0.02).minimize(totalLoss)
trainingAlg = tf.train.AdagradOptimizer(0.001).minimize(totalLoss)

# and train!!
# with tf.Session() as sess:
#     #
#     # initialize everything
#     sess.run(tf.global_variables_initializer())
#     #
#     global_step = 0
#
#     # numTrainingIters = int(len(y_train) / batchSize)
#     # and run the training iters
#     for epoch in range(numTrainingIters):
#         #
#         # get some data
#         x, y = generateBatchData(x_train, y_train)
#
#         #
#         # do the training epoch
#         _totalLoss, _trainingAlg, _predictions, _outputs = sess.run(
#             [totalLoss, trainingAlg, predictions, outputs],
#             feed_dict={
#                 inputX: x,
#                 inputY: y,
#             })
#         # just FYI, compute the number of correct predictions
#         numCorrect = 0
#         for i in range(len(y)):
#             maxPos = -1
#             maxVal = 0.0
#             for j in range(numClasses):
#                 if maxVal < _predictions[i][j]:
#                     maxVal = _predictions[i][j]
#                     maxPos = j
#             if maxPos == y[i]:
#                 numCorrect = numCorrect + 1
#         #
#         # print out to the screen
#         print("Step", epoch, "Loss", _totalLoss, "Correct", numCorrect, "out of", batchSize)
#     numCorrect = 0
#     lst_outputs = []
#     lst_y = []
#     for i in range(30):
#         x = np.stack(np.array((x_test[i])) for i in range(100 * i, 100 * (i + 1)))
#         # and stack all of the labels into a vector of labels
#         y = np.stack(np.array((y_test[i])) for i in range(100 * i, 100 * (i + 1)))
#         _totalLoss, _predictions, _outputs = sess.run(
#             [totalLoss, predictions, outputs],
#             feed_dict={
#                 inputX: x,
#                 inputY: y,
#             })
#         lst_outputs.append(_outputs)
#         lst_y.append(y)
#         for i in range(len(y)):
#             maxPos = -1
#             maxVal = 0.0
#             for j in range(numClasses):
#                 if maxVal < _predictions[i][j]:
#                     maxVal = _predictions[i][j]
#                     maxPos = j
#             if maxPos == y[i]:
#                 numCorrect = numCorrect + 1
#     total_outputs = np.concatenate(lst_outputs)
#     total_y = np.concatenate(lst_y)
#     res_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=total_outputs, labels=total_y)
#     final_loss = tf.reduce_mean(res_losses)
#     _resLoss = sess.run(final_loss)
#     print(
#     "Loss for 3000 randomly chosen documents is " + str(_resLoss) + ", number of correct labels is " + str(numCorrect),
#     "out of 3000")