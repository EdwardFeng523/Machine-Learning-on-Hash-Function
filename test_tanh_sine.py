import tensorflow as tf
import random
import numpy as np

batchSize = 100

numClasses = 2

hiddenUnits = 50

numTrainingIters = 20000

TABLE_SIZE = 10000



def generateBatchData(x, y):
    #
    # randomly sample batchSize lines of text
    myInts = np.random.random_integers(0, len(x) - 1, batchSize)
    #
    # stack all of the text into a matrix of one-hot characters
    x_train = np.stack(np.array((x[i])) for i in myInts.flat)

    #
    # and stack all of the labels into a vector of labels
    y_train = np.stack(np.array((y[j])) for j in myInts.flat)
    #
    # return the pair
    return (x_train, y_train)

def h(x):
    # return ((37 * x + 47) % 2038072819) % TABLE_SIZE
    return (37 * x + 47)% TABLE_SIZE

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


# cnt = 0
# for i in range(TABLE_SIZE):
#     if table[i] == 1:
#         cnt += 1
# print "total 1", cnt

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
            x_set.add(datum)
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





# print x_train
# print y_train
# print x_test
# print y_test

## Tuning the training data to be balanced

with tf.name_scope('input_X'):
    # a sequence of numbers representing the items to test the cache
    inputX = tf.placeholder(tf.float32, [batchSize,])

with tf.name_scope('input_labels'):
    # inputy is a sequence of zeroes and ones
    inputY = tf.placeholder(tf.int32, [batchSize,])

with tf.name_scope('hidden_weights'):
    # the weight matrix that maps the inputs to hiddden layer
    W = tf.Variable(np.random.normal(0, 0.05, (1, hiddenUnits)), dtype=tf.float32)

with tf.name_scope('hidden_weights_2'):
    # the weight matrix that maps the inputs to hiddden layer
    W2 = tf.Variable(np.random.normal(0, 0.05, (hiddenUnits, hiddenUnits)), dtype=tf.float32)

with tf.name_scope('hidden_biases_2'):
    # biaes for the hidden values
    b = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

with tf.name_scope('hidden_biases'):
    # biaes for the hidden values
    b2 = tf.Variable(np.zeros((1, hiddenUnits)), dtype=tf.float32)

with tf.name_scope('final_weights'):
    # weights and bias for the final classification
    W3 = tf.Variable(np.random.normal (0, 0.05, (hiddenUnits, numClasses)), dtype=tf.float32)
with tf.name_scope('final_biases'):
    b3 = tf.Variable(np.zeros((1,numClasses)), dtype=tf.float32)

hiddenLayer = tf.tanh(tf.matmul(tf.reshape(inputX, [100, 1]), W) + b)

hiddenLayer2 = tf.math.sin(tf.matmul(hiddenLayer, W2) + b2)

outputs = tf.matmul(hiddenLayer2, W3) + b3

predictions = tf.nn.softmax(outputs)

# compute the loss
with tf.name_scope('x_entropy'):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=inputY)
with tf.name_scope('total_loss'):
    totalLoss = tf.reduce_mean(losses)
tf.summary.scalar('loss', totalLoss)


# use gradient descent to train
# trainingAlg = tf.train.GradientDescentOptimizer(0.02).minimize(totalLoss)
trainingAlg = tf.train.AdagradOptimizer(0.1).minimize(totalLoss)
merged = tf.summary.merge_all()

# and train!!
with tf.Session() as sess:
    #
    # initialize everything
    train_writer = tf.summary.FileWriter('./logs_tanh_sine/', sess.graph)
    x_train, y_train, x_test, y_test = generate_random_data()
    sess.run(tf.global_variables_initializer())
    #
    global_step = 0

    # numTrainingIters = int(len(y_train) / batchSize)
    # and run the training iters
    for epoch in range(numTrainingIters):
        #
        # get some data
        x, y = generateBatchData(x_train, y_train)
        #
        #
        # do the training epoch
        _merged, _totalLoss, _trainingAlg, _predictions, _outputs = sess.run(
            [merged, totalLoss, trainingAlg, predictions, outputs],
            feed_dict={
                inputX: x,
                inputY: y
            })
        train_writer.add_summary(_merged, epoch)
        # just FYI, compute the number of correct predictions
        numCorrect = 0
        for i in range(len(y)):
            maxPos = -1
            maxVal = 0.0
            for j in range(numClasses):
                if maxVal < _predictions[i][j]:
                    maxVal = _predictions[i][j]
                    maxPos = j
            if maxPos == y[i]:
                numCorrect = numCorrect + 1
        #
        # print out to the screen
        print("Step", epoch, "Loss", _totalLoss, "Correct", numCorrect, "out of", batchSize)
    numCorrect = 0
    lst_outputs = []
    lst_y = []
    for i in range(len(x_test) / 100):
        x = np.stack(np.array((x_test[i])) for i in range(100 * i, 100 * (i + 1)))
        # and stack all of the labels into a vector of labels
        y = np.stack(np.array((y_test[i])) for i in range(100 * i, 100 * (i + 1)))
        _totalLoss, _predictions, _outputs = sess.run(
            [totalLoss, predictions, outputs],
            feed_dict={
                inputX: x,
                inputY: y
            })
        lst_outputs.append(_outputs)
        lst_y.append(y)
        for i in range(len(y)):
            maxPos = -1
            maxVal = 0.0
            for j in range(numClasses):
                if maxVal < _predictions[i][j]:
                    maxVal = _predictions[i][j]
                    maxPos = j
            if maxPos == y[i]:
                numCorrect = numCorrect + 1
    total_outputs = np.concatenate(lst_outputs)
    total_y = np.concatenate(lst_y)
    res_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=total_outputs, labels=total_y)
    final_loss = tf.reduce_mean(res_losses)
    _resLoss = sess.run(final_loss)
    print(
    "Loss for test set is " + str(_resLoss) + ", number of correct labels is " + str(numCorrect),
    "out of", len(x_test))
