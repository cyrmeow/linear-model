import scipy.io as sio  # reading in matlab files
import numpy as np
import random


starplus = sio.loadmat("data-starplus-04847-v7.mat")

###########################################
metadata = starplus['meta'][0,0]
#meta.study gives the name of the fMRI study
#meta.subject gives the identifier for the human subject
#meta.ntrials gives the number of trials in this dataset
#meta.nsnapshots gives the total number of images in the dataset
#meta.nvoxels gives the number of voxels (3D pixels) in each image
#meta.dimx gives the maximum x coordinate in the brain image. The minimum x coordinate is x=1. meta.dimy and meta.dimz give the same information for the y and z coordinates.
#meta.colToCoord(v,:) gives the geometric coordinate (x,y,z) of the voxel corresponding to column v in the data
#meta.coordToCol(x,y,z) gives the column index (within the data) of the voxel whose coordinate is (x,y,z)
#meta.rois is a struct array defining a few dozen anatomically defined Regions Of Interest (ROIs) in the brain. Each element of the struct array defines on of the ROIs, and has three fields: "name" which gives the ROI name (e.g., 'LIFG'), "coords" which gives the xyz coordinates of each voxel in that ROI, and "columns" which gives the column index of each voxel in that ROI.
#meta.colToROI{v} gives the ROI of the voxel corresponding to column v in the data.
study      = metadata['study']
subject    = metadata['subject']
ntrials    = metadata['ntrials'][0][0]
nsnapshots = metadata['nsnapshots'][0][0]
dimx       = metadata['dimx'][0][0]
colToCoord = metadata['colToCoord']
coordToCol = metadata['coordToCol']
rois       = metadata['rois']
colToROI   = metadata['colToROI']
###########################################

###########################################
info = starplus['info'][0]
#info: This variable defines the experiment in terms of a sequence of 'trials'. 'info' is a 1x54 struct array, describing the 54 time intervals, or trials. Most of these time intervals correspond to trials during which the subject views a single picture and a single sentence, and presses a button to indicate whether the sentence correctly describes the picture. Other time intervals correspond to rest periods. The relevant fields of info are illustrated in the following example:
#info(18) mint: 894 maxt: 948 cond: 2 firstStimulus: 'P' sentence: ''It is true that the star is below the plus.'' sentenceRel: 'below' sentenceSym1: 'star' sentenceSym2: 'plus' img: sap actionAnswer: 0 actionRT: 3613
#info.mint gives the time of the first image in the interval (the minimum time)
#info.maxt gives the time of the last image in the interval (the maximum time)
#info.cond has possible values 0,1,2,3. Cond=0 indicates the data in this segment should be ignored. Cond=1 indicates the segment is a rest, or fixation interval. Cond=2 indicates the interval is a sentence/picture trial in which the sentence is not negated. Cond=3 indicates the interval is a sentence/picture trial in which the sentence is negated.
#info.firstStimulus: is either 'P' or 'S' indicating whether this trail was obtained during the session is which Pictures were presented before sentences, or during the session in which Sentences were presented before pictures. The first 27 trials have firstStimulus='P', the remained have firstStimulus='S'. Note this value is present even for trials that are rest trials. You can pick out the trials for which sentences and pictures were presented by selecting just the trials trials with info.cond=2 or info.cond=3.
#info.sentence gives the sentence presented during this trial. If none, the value is '' (the empty string). The fields info.sentenceSym1, info.sentenceSym2, and info.sentenceRel describe the two symbols mentioned in the sentence, and the relation between them.
#info.img describes the image presented during this trial. For example, 'sap' means the image contained a 'star above plus'. Each image has two tokens, where one is above the other. The possible tokens are star (s), plus (p), and dollar (d).
#info.actionAnswer: has values -1 or 0. A value of 0 indicates the subject is expected to press the answer button during this trial (either the 'yes' or 'no' button to indicate whether the sentence correctly describes the picture). A value of -1 indicates it is inappropriate for the subject to press the answer button during this trial (i.e., it is a rest, or fixation trial).
#info.actionRT: gives the reaction time of the subject, measured as the time at which they pressed the answer button, minus the time at which the second stimulus was presented. Time is in milliseconds. If the subject did not press the button at all, the value is 0.
###########################################

###########################################
data = starplus['data']
#data: This variable contains the raw observed data. The fMRI data is a sequence of images collected over time, one image each 500 msec. The data structure 'data' is a [54x1] cell array, with one cell per 'trial' in the experiment. Each element in this cell array is an NxV array of observed fMRI activations. The element data{x}(t,v) gives the fMRI observation at voxel v, at time t within trial x. Here t is the within-trial time, ranging from 1 to info(x).len. The full image at time t within trial x is given by data{x}(t,:).
#Note the absolute time for the first image within trial x is given by info(x).mint.
###########################################

def HingeLoss(X, Y, W, lmda):
    #TODO: Compute (regularized) Hinge Loss
    loss_l = [np.max([0, -Y[l] * np.dot(W, X[l])]) for l in range(0, X.shape[0])]
    loss = np.sum(loss_l) + lmda * np.sum(W ** 2)
    return loss

def SgdHinge(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1])
    #TODO: implement stochastic (sub) gradient descent with the hinge loss function
    loss = HingeLoss(X, Y, W, lmda) + 1
    loss_new = HingeLoss(X, Y, W, lmda)
    W_new = W
    i = 0
    np.random.seed(1)
    print loss_new
    while i < maxIter and np.absolute(loss_new - loss) > 0.001:
        # loss = loss_new
        j = 0
        for j in range(0, X.shape[0]):
            loss = loss_new
            l = np.random.random_integers(0, X.shape[0] - 1)
            gradient = float(2) / X.shape[0] * lmda * W
            if 1 - Y[l] * np.dot(W, X[l]) > 0:
                gradient = -Y[l] * X[l] + gradient
            W = W - learningRate * gradient
            loss_new = HingeLoss(X, Y, W, lmda)
            if np.absolute(loss_new - loss) <= 0.001:
                break
        # print loss_new
        if j < X.shape[0] - 1:
            break
        i = i + 1
    return W

def LogisticLoss(X, Y, W, lmda):
    #TODO: Compute (regularized) Logistic Loss
    print "W = %s" % W
    reg = np.sum(W ** 2)
    print "reg term = %s" % reg
    dot_product = [np.dot(W, xl) for xl in X]
    print "dot product = %s" % dot_product
    power = -Y * dot_product
    print "powers = %s" % power
    larger = [np.max([i, 0]) for i in power]
    loss = np.sum(larger + np.log(np.exp(np.zeros(len(larger))-larger) + np.exp(power - larger))) + lmda * reg
    print "loss = %s" % loss
    return loss

def SgdLogistic(X, Y, maxIter, learningRate, lmda):
    W = np.zeros(X.shape[1])
    #TODO: implement stochastic gradient descent using the logistic loss function
    loss = LogisticLoss(X, Y, W, lmda) + 1
    loss_new = LogisticLoss(X, Y, W, lmda)
    i = 0
    print loss_new
    np.random.seed(1)
    while i < maxIter and np.absolute(loss_new - loss) > 0.001:
        loss = loss_new
        j = 0    
        for j in range(0, X.shape[0]):
            loss = loss_new
            l = np.random.random_integers(0, X.shape[0] - 1)
            power = -Y[l] * np.dot(W, X[l])
            larger = np.max([power, 0])
            temp = np.exp(power - larger) / (np.exp(-larger) + np.exp(power - larger))
            gradient = -Y[l] * X[l] * temp + (2 * lmda * W) / X.shape[0] 
            W = W - learningRate * gradient
            loss_new = LogisticLoss(X, Y, W, lmda)
            if np.absolute(loss_new - loss) <= 0.001:
                break
        # print W_new
        print loss_new
        if j < X.shape[0] - 1:
            break
        i = i + 1
    return W

def crossValidation(X, Y, SGD, lmda, learningRate, maxIter=100, sample=range(20)):
    #Leave one out cross validation accuracy
    nCorrect   = 0.
    nIncorrect = 0.
    
    for i in sample:
        print "CROSS VALIDATION %s" % i
        
        training_indices = [j for j in range(X.shape[0]) if j != i]
        W = SGD(X[training_indices,], Y[training_indices,], maxIter=maxIter, lmda=lmda, learningRate=learningRate)
        print W
        y_hat = np.sign(X[i,].dot(W))

        if y_hat == Y[i]:
            nCorrect += 1
        else:
            nIncorrect += 1

    return nCorrect / (nCorrect + nIncorrect)

def Accuracy(X, Y, W):
    Y_hat = np.sign(X.dot(W))
    correct = (Y_hat == Y)
    return float(sum(correct)) / len(correct)

def main():
    maxFeatures =  max([data[i][0].flatten().shape[0] for i in range(data.shape[0])])

    #Inputs
    X = np.zeros((ntrials, maxFeatures+1))
    for i in range(data.shape[0]):
        f = data[i][0].flatten()
        X[i,:f.shape[0]] = f
        X[i,f.shape[0]]  = 1     #Bias

    #Outputs (+1 = Picture, -1 = Sentence)
    Y = np.ones(ntrials)
    Y[np.array([info[i]['firstStimulus'][0] != 'P' for i in range(ntrials)])] = -1

    #Randomly permute the data
    np.random.seed(1)      #Seed the random number generator to preserve the dev/test split
    permutation = np.random.permutation(ntrials)
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation,]
    Y = Y[permutation,]

    #Cross validation
    #Development
    # print "Logistic Loss: \n"
    # for l in [0.3, 0.1, 1]:
    #     for eta in [0.0001, 0.001, 0.01, 0.1]:
    #         print "lambda = %s\n" % l
    #         print "learningRate = %s\n" % eta
    #         print "Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=l, learningRate=eta, sample=range(20))print "Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=1, learningRate=0.1, sample=range(20))

    # print "Hinge Loss: \n"
    # for l in [0.3, 0.1, 1]:
    #     for eta in [0.0001, 0.001, 0.01, 0.1]:
    #         print "lambda = %s\n" % l
    #         print "learningRate = %s\n" % eta
    #         print "Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=l, learningRate=eta, sample=range(20))
    
    # for l in [0.3, 0.1, 1]:
    #     for eta in [0.01, 0.1]:
    #         print "lambda = %s\n" % l
    #         print "learningRate = %s\n" % eta
    #         print "Accuracy (Logistic Loss):\t%s" % crossValidation(X, Y, SgdLogistic, maxIter=100, lmda=l, learningRate=eta, sample=range(20,X.shape[0]))
    # for l in [0.3, 1]: 
    #     print "lambda = %s\n" % l
    #     print "learningRate = 0.1\n"
    #     print "Accuracy (Hinge Loss):\t%s" % crossValidation(X, Y, SgdHinge, maxIter=100, lmda=l, learningRate=0.1, sample=range(20,X.shape[0]))

    # Inspecting the Modle's Parameters
    W = SgdLogistic(X, Y, maxIter=100, lmda=1, learningRate=0.1)
    print len(W)
    important_roi = []
    for i in range(0, 55):
        W_cols = np.asarray([W[i] for i in np.arange(0, 4698) + i * 4698])
        ROI = np.unique(colToROI)
        W_ROI = [[np.average(np.abs(W_cols[rois['columns'][0][i][0] - 1])), rois['name'][0][i][0]] for i in range(0, rois.shape[1])]
        max_index = np.argmax([i[0] for i in W_ROI])
        important_roi.append(W_ROI[max_index][1])
    print max(important_roi, key = important_roi.count)

if __name__ == "__main__":
    main()
