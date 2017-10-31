from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import glob
import statistics
import theano.tensor as T
#import scipy.optimize as optimization
def conv1d_sc(input, filters, image_shape=None, filter_shape=None,
              border_mode='valid', subsample=(1,), filter_flip=True):
    """
    using conv2d with a single input channel
    """
    if border_mode not in ('valid', 0, (0,)):
        raise RuntimeError("Unsupported border_mode for conv1d_sc: "
                           "%s" % border_mode)

    if image_shape is None:
        image_shape_sc = None
    else:
        # (b, c, i0) to (b, 1, c, i0)
        image_shape_sc = (image_shape[0], 1, image_shape[1], image_shape[2])

    if filter_shape is None:
        filter_shape_sc = None
    else:
        filter_shape_sc = (filter_shape[0], 1, filter_shape[1],
                           filter_shape[2])

    input_sc = input.dimshuffle(0, 'x', 1, 2)
    # We need to flip the channels dimension because it will be convolved over.
    filters_sc = filters.dimshuffle(0, 'x', 1, 2)[:, :, ::-1, :]

    conved = T.nnet.conv2d(input_sc, filters_sc, image_shape_sc,
                           filter_shape_sc, subsample=(1, subsample[0]),
                           filter_flip=filter_flip)
    return conved[:, :, 0, :]  # drop the unused dimension




def conv1d_mc0(input, filters, image_shape=None, filter_shape=None,
               border_mode='valid', subsample=(1,), filter_flip=True):
    """
    using conv2d with width == 1
    """
    if image_shape is None:
        image_shape_mc0 = None
    else:
        # (b, c, i0) to (b, c, 1, i0)
        image_shape_mc0 = (image_shape[0], image_shape[1], 1, image_shape[2])

    if filter_shape is None:
        filter_shape_mc0 = None
    else:
        filter_shape_mc0 = (filter_shape[0], filter_shape[1], 1,
                            filter_shape[2])

    if isinstance(border_mode, tuple):
        (border_mode,) = border_mode
    if isinstance(border_mode, int):
        border_mode = (0, border_mode)

    input_mc0 = input.dimshuffle(0, 1, 'x', 2)
    filters_mc0 = filters.dimshuffle(0, 1, 'x', 2)

    conved = T.nnet.conv2d(
        input_mc0, filters_mc0, image_shape_mc0, filter_shape_mc0,
        subsample=(1, subsample[0]), border_mode=border_mode,
        filter_flip=filter_flip)
    return conved[:, :, 0, :]  # drop the unused dimension



def inputdata(filepath):
    filelist = glob.glob(filepath)
    
    trainingdata = []
    for afile in filelist:
        f = open(afile, "r")
        for s_line in iter(f):
            s_line = s_line.upper()
            sz_dga_matrix = []
            for c in s_line.rstrip('\n').split(',')[0].split('.')[0]:
                sz_dga_matrix.append(ord(c))
            trainingdata.append(np.asarray(sz_dga_matrix))
    return trainingdata

def error_correction(x):
    return x*1.0*(1.0-x)
    
def func(x,b,w):
    return w*x+b

#def lsqfunction(trainingdata,predctiondata,func,weights):
#    return optimization.leastsq(func,weights,args=(trainingdata,predctiondata))

def testfucntion(testdata,weights,bias):
    for x in range(len(testdata)-1):
            
            if(len(testdata[x])>len(weights)):
                for i in range(len(testdata[x])-len(weights)):
                    weights = np.append(weights,0)
                predictresult = testdata[x]*weights+bias
            elif(len(weights)>len(traindata[x])):
                for i in range(len(weights)-len(testdata[x])):
                    traindata[x] = np.append(traindata[x],0)
                predictresult = testdata[x]*weights+bias
                
            else:
                predictresult = testdata[x]*weights+bias
            predictresultlist.append(predictresult)
    return predictresultlist



def learningmodel(traindata,length,learningrate):
    print("training....")
    #randomize the initial parameters
    weights = 2*np.random.random(len(traindata[0])) - 1
    #print(len(traindata[0]))
    bias = 0
    filt = 2*np.random.random(5)-1
    print(filt)
    #error=[]
    #delta = []
    print(bias)
    predictresultlist = []
    
    #presult = traindata[0]*weights+bias
    #e0 = presult - traindata[0]
    c = 0
    for le in range(len(traindata)):
        try:
            presult = traindata[le]*weights+bias
            e0 = abs(presult - traindata[le])
            #print(presult)
            for item in e0:
                #print (item)
                if item < 0.5:
                    c += 1
            
            #print(count)
            #print(len(e0))
        except ValueError:
            pass
    tlitems = 15*len(traindata)
    print(c)
    print("random accuracy:")
    print(c*1.0/tlitems)
    #print(sum(e0*e0)/10)

    #the training iteration
    for iter in range(100):
        error=[]
        delta = []
    #training predction
        for x in range(len(traindata)-1):
            
            if(len(traindata[x])>len(weights)):
                for i in range(len(traindata[x])-len(weights)):
                    weights = np.append(weights,0)
                predictresult = traindata[x]*weights+bias
            elif(len(weights)>len(traindata[x])):
                for i in range(len(weights)-len(traindata[x])):
                    traindata[x] = np.append(traindata[x],0)
                predictresult = traindata[x]*weights+bias
                #predictresult = traindata[x]*weights+bias
            else:
                predictresult = np.longdouble(traindata[x])*np.longdouble(weights)+bias
            predictresultlist.append(predictresult)

            #for y in range(length):
            
            if len(predictresult)==len(traindata[x+1]):
                temerror = predictresult - traindata[x+1]
                error.append(temerror)
                #delta.append(error[y+1] * error_correction(predictresult[-(y+2)]))
        #training correction
        #lq = lsqfunction(traindata,predictresultlist,func,weights)
        #print(lq)
        for y in range(len(error)):
                correction = -1.0*learningrate*error[y]/(traindata[y]+0.01)
                delta.append(correction)
                #updates weights
                weights += correction
        #print(weights)
    #calcualte lsq
    count = 0
    for le in range(len(traindata)):
        try:
            presult = traindata[le]*weights+bias
            e0 = abs(presult - traindata[le])
            #print(presult)
            for item in e0:
                #print (item)
                if item < 0.5:
                    count += 1
            
            #print(count)
            #print(len(e0))
        except ValueError:
            pass
    totalitems = 15*len(traindata)
    print(sum(abs(e0))) 
    print("correct prediction.")
    print(count)   
    print(totalitems)
    print("accuracy:")
    print(count*1.0/totalitems)
    return weights    
                
        #print(temerror)
        
        #for z in range(layers):
        #    weights[z] += np.dot(predictresult[z].T,delta[-(z+1)])
        #    bias[z] += np.mean(delta[-(z+1)])
    
        
if __name__ == "__main__":
    
    tdata = inputdata("./*.txt")
    w = learningmodel(tdata,5,0.01)
    print("trained model")
    print(w)
    
    
