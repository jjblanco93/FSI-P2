import gzip
import cPickle
import numpy as np
import matplotlib

matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

y = tf.nn.softmax(tf.matmul(x, W1) + b1)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


def print_results(mode, error, batch_xs, batch_ys, epoch_number=1):
    print mode, " epoch #:", epoch_number, "Error: ", error
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r


print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
training_errors = []
validation_errors = []
test_errors = []
last_validation_error = 1
epoch = 0
validation_error = 0.1
difference = 100.0

while difference > 0.1:
    epoch += 1
    for jj in xrange(len(train_x) / batch_size):
        batch_training_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_training_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_training_xs, y_: batch_training_ys})

    training_error = sess.run(loss, feed_dict={x: batch_training_xs, y_: batch_training_ys})
    training_errors.append(training_error)

    validation_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    validation_errors.append(validation_error)
    
    if epoch > 1:
        difference = validation_errors[-2] - validation_error
    print "validation error", validation_error

# ---------------- Visualizing some element of the MNIST dataset --------------

print "----------------------"
print "   Start testing...  "
print "----------------------"


test_error = sess.run(loss, feed_dict={x: test_x, y_: test_y})

print_results(mode="Testing", error=test_error, batch_xs=test_x, batch_ys=test_y)

error = 0
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    if np.argmax(b) != np.argmax(r):
        error += 1
    print b, "-->", r

success = 100 - (error * 100 / 10000)

error = error * 100 / 10000
print "----------------------------------------------------------------------------------"
print "Error:", error, "%"
print "----------------------------------------------------------------------------------------"
print "Success:", success, "%"
print "----------------------------------------------------------------------------------------"

# print "   Testing finished: error ", test_error, ", last validation error ", validation_errors[-1]

# print "----------------------------------------------------------------------------------------"

plt.ylabel('Errors')
plt.xlabel('Epochs')
training_line, = plt.plot(training_errors)
validation_line, = plt.plot(validation_errors)
plt.legend(handles=[training_line, validation_line],
           labels=["Training errors", "Validation errors"])
plt.savefig('mnist.png')