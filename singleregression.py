'''
A linear regression learning algorithm example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import glob
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
import sklearn

alphacheck = {
'A':1,
'B':2,
'C':3,
'D':4,
'E':5,
'F':6,
'G':7,
'H':8,
'I':9,
'J':10,
'K':11,
'L':12,
'M':13,
'N':14,
'O':15,
'P':16,
'Q':17,
'R':18,
'S':19,
'T':20,
'U':21,
'V':22,
'W':23,
'X':24,
'Y':25,
'Z':26,
'0':100,
'1':101,
'2':102,
'3':103,
'4':104,
'5':105,
'6':106,
'7':107,
'8':108,
'9':109,
'-':200
}

def inputdata(filepath):
    filelist = glob.glob(filepath)
    
    trainingdata = []
    for afile in filelist:
        f = open(afile, "r")
        for s_line in iter(f):
            s_line = s_line.upper()
            sz_dga_matrix = []
            for c in s_line.rstrip('\n').split(',')[0].split('.')[0]:
                sz_dga_matrix.append(alphacheck[c])
            trainingdata.append(numpy.asarray(sz_dga_matrix))
    return trainingdata

tdata = inputdata("./*.txt")
x1 = tdata[:-80]
y1 = tdata[1:-79]
x2 = tdata[-80:-1]
y2 = tdata[-79:]

bigX = []
bigY = []
TX = []
TY = []
for xx in x2:
    #print(xx)
    TX.append(xx[11])
for yy in y2:
    #print(yy)
    TY.append(yy[11])


for xx in x1:
    print(xx)
    bigX.append(xx[11])
for yy in y1:
    #print(yy)
    bigY.append(yy[11])

#print(bigX)
#print(bigY)

# Parameters
learning_rate = 0.001
training_epochs = 100
display_step = 50

# Training Data
train_X = numpy.asarray(bigX)
train_Y = numpy.asarray(bigY)
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray(TX)
    test_Y = numpy.asarray(TY)

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

