from __future__ import print_function
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
import time
rng = np.random


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



def inputdata(filename):   
    trainingdata = []
    #for afile in filelist:
    f = open(filename, "r")
    for s_line in iter(f):
            s_line = s_line.upper()
            sz_dga_matrix = []
            for c in s_line.rstrip('\n').split(',')[0].split('.')[0]:
                sz_dga_matrix.append(alphacheck[c])
            sz_dga_matrix = sz_dga_matrix[:12]
            trainingdata.append(np.asarray(sz_dga_matrix))
    return np.asarray(trainingdata)

def inputlabel(labelfile):
    f = open(labelfile, "r")
    label = []
    for s_line in iter(f):
        ll = s_line.split(',')
        ls = []
        for l in ll:
            ls.append(float(l))
        label.append(ls)
    return np.asarray(label)


def weight_variable(shape):
    initial = truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    
    pass
    
# Parameters
learning_rate = 0.01
training_epochs = 2000
display_step = 50
tdata = inputdata("CryptoLockerData.txt")
tlabel = inputlabel("DGAlabel2nd.csv")

bigX = tdata[:-1]
bigY = tlabel[1:]
n_samples = bigX.shape[0]
X = tf.placeholder("float",[None,12])
Y_ = tf.placeholder("float",[None,26])
W = tf.Variable(tf.zeros([12,26]), name="weight")
b = tf.Variable(tf.zeros([26]), name="bias")
pred = tf.nn.softmax(tf.matmul(X, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_*tf.log(pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


init = tf.global_variables_initializer()
start = time.time()
for epoch in range(training_epochs):
    sess.run(train_step, feed_dict = {X: bigX, Y_: bigY})
end = time.time()
print("training time:")
print(end - start)


correctcount = tf.equal(tf.argmax(pred,1), tf.argmax(Y_,1))

accuracy = tf.reduce_mean(tf.cast(correctcount, tf.float32))

    #print("model")
    #print(sess.run(W))
    #print(sess.run(b))
    #saver = tf.train.Saver()
    #save_path = saver.save(sess, "./model.ckpt")
    #print("model saved!")
print("accuracy is:")
start = time.time()
print(sess.run(accuracy, feed_dict = {X: bigX, Y_: bigY} ))
end = time.time()
print("testing time:")
print(end - start)
exit(0)
# Training Data
#train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
#                       7.042,10.791,5.313,7.997,5.654,9.27,3.1])
#train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
#                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])





    # Set model weights






    # Gradient descent
    #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default


    # Initialize the variables (i.e. assign their default value)








