from __future__ import division

import matplotlib.pyplot as plt
import numpy.matlib as matlib
from scipy.stats import multivariate_normal
import numpy as np

'''
Notes for ML1003:

This is support code provided for the Bayesian Regression Problems.
The goal of this problem is to have you explore Bayesian Regression,
as described in Lectur 11.b slides 13-25.

A few things to note about this code:
    - We strongly encourage you to review this support code prior to
      completing "problem.py"
    - For Problem (b), you are asked to generate plots for three
      values of sigma_squared. We suggest you savid the plot generated
      by make_plots (instead of simply calling plt.show)
'''

# Notes from intial author (prior to some)
# translating the implementation at
# https://github.com/probml/pmtk3/blob/master/demos/bayesLinRegDemo2d.m
# into python
#
# Bayesian inference for simple linear regression with known noise variance
# The goal is to reproduce fig 3.7 from Bishop's book
# We fit the linear model f(x,w) = w0 + w1 x and plot the posterior over w.
# This file is from pmtk3.googlecode.com
# Given the mean = priorMu and covarianceMatrix = priorSigma of a prior
# Gaussian distribution over regression parameters; observed data, xtrain
# and ytrain; and the likelihood precision, generate the posterior
# distribution, postW via Bayesian updating and return the updated values
# for mu and sigma. xtrain is a design matrix whose first column is the all
# ones vector.

def generateData(dataSize, noiseParams, actual_weights):
    # x1: from [0,1) to [-1,1)
    x1 = -1 + 2 * np.random.rand(dataSize, 1)
    # appending the bias term
    xtrain = np.matrix(np.c_[np.ones((dataSize, 1)), x1])
    # random noise
    noise = np.matrix(np.random.normal(
                            noiseParams["mean"],
                            noiseParams["var"],
                            (dataSize, 1)))

    ytrain = (xtrain * actual_weights) + noise

    return xtrain, ytrain

def make_plots(actual_weights, xtrain, ytrain, likelihood_var, prior, likelihoodFunc, getPosteriorParams, getPredictiveParams):

    # #setup for plotting
    #
    showProgressTillDataRows = [1, 2, 10, -1]
    numRows = 1 + len(showProgressTillDataRows)
    numCols = 4
    plt.figure(figsize=(10,10))
    plt.subplots_adjust(hspace=.8, wspace=.8)

    plotWithoutSeeingData(prior, numRows, numCols)

    # see data for as many rounds as specified and plot
    for roundNum, rowNum in enumerate(showProgressTillDataRows):
        current_row = roundNum + 1
        first_column_pos = (current_row * numCols) + 1

        # #plot likelihood on latest point
        plt.subplot(numRows, numCols, first_column_pos)


        likelihoodFunc_with_data = lambda W: likelihoodFunc(W,
                                                      xtrain[:rowNum,],
                                                      ytrain[:rowNum],
                                                      likelihood_var)
        contourPlot(likelihoodFunc_with_data, actual_weights)

        # plot updated posterior on points seen till now
        x_seen = xtrain[:rowNum]
        y_seen = ytrain[:rowNum]
        mu, cov = getPosteriorParams(x_seen, y_seen,
                                      prior, likelihood_var)
        posteriorDistr = multivariate_normal(mu.T.tolist()[0], cov)
        posteriorFunc = lambda x: posteriorDistr.pdf(x)
        plt.subplot(numRows, numCols, first_column_pos + 1)
        contourPlot(posteriorFunc, actual_weights)

        # plot lines
        dataSeen = np.c_[x_seen[:, 1], y_seen]
        plt.subplot(numRows, numCols, first_column_pos + 2)
        plotSampleLines(mu, cov, dataPoints=dataSeen)

        # plot predictive
        plt.subplot(numRows, numCols, first_column_pos + 3)
        postMean, postVar = getPosteriorParams(x_seen, y_seen, prior)
        plotPredictiveDistribution(getPredictiveParams, postMean, postVar)

    # #show the final plot
    plt.show()

def plotWithoutSeeingData(prior, numRows, numCols):

    #Blank likelihood
    plt.subplot(numRows, numCols, 1, axisbg='grey')
    plt.title("Likelihood")
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-0.9, 0.9])
    plt.ylim([-0.9, 0.9])

    #Prior
    priorDistribution = multivariate_normal(mean=prior["mean"].T.tolist()[0],
        cov=prior["var"])
    priorFunc = lambda x:priorDistribution.pdf(x)
    plt.subplot(numRows, numCols, 2)
    plt.title("Prior/Posterior")
    contourPlot(priorFunc)

    # Plot initially valid lines (no data seen)
    plt.subplot(numRows, numCols, 3)
    plt.title("Data Space")
    plotSampleLines(prior["mean"], prior["var"])

    # Blank predictive
    plt.subplot(numRows, numCols, 4, axisbg='grey')
    plt.title('Predictive Distribution')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("")
    plt.ylabel("")

def contourPlot(distributionFunc, actualWeights=[]):

    stepSize = 0.05
    array = np.arange(-1, 1, stepSize)
    x, y_train = np.meshgrid(array, array)

    length = x.shape[0] * x.shape[1]
    x_flat = x.reshape((length, 1))
    y_flat = y_train.reshape((length, 1))
    contourPoints = np.c_[x_flat, y_flat]

    values = map(distributionFunc, contourPoints)
    values = np.array(values).reshape(x.shape)

    plt.contourf(x, y_train, values)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.xticks([-0.5, 0, 0.5])
    plt.yticks([-0.5, 0, 0.5])
    plt.xlim([-0.9, 0.9])
    plt.ylim([-0.9, 0.9])

    if(len(actualWeights) == 2):
        plt.plot(float(actualWeights[0]), float(actualWeights[1]),
                 "*k", ms=5)

# Plot the specified number of lines of the form y_train = w0 + w1*x in [-1,1]x[-1,1] by
# drawing w0, w1 from a bivariate normal distribution with specified values
# for mu = mean and sigma = covariance Matrix. Also plot the data points as
# circles.
def plotSampleLines(mean, variance,
                    numberOfLines=6,
                    dataPoints=np.empty((0, 0))):
    stepSize = 0.05
    # generate and plot lines
    for round in range(1, numberOfLines):
        weights = np.matrix(np.random.multivariate_normal(mean.T.tolist()[0], variance)).T
        x1 = np.arange(-1, 1, stepSize)
        x = np.matrix(np.c_[np.ones((len(x1), 1)), x1])
        y_train = x * weights

        plt.plot(x1, y_train)

    # markings
    plt.xticks([-1, 0, 1])
    plt.yticks([-1, 0, 1])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")

    # plot data points if given
    if(dataPoints.size):
        plt.plot(dataPoints[:, 0], dataPoints[:, 1],
                 "co")

def plotPredictiveDistribution(getPredictiveParams,postMean, postVar):
    stepSize = 0.05
    x = np.arange(-1, 1, stepSize)
    x = np.matrix(np.c_[np.ones((len(x), 1)), x])
    predMeans = np.zeros(x.shape[0])
    predStds = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        predMeans[i], predStds[i] = getPredictiveParams(x[i,].T,
                                                        postMean,
                                                        postVar)
    predStds = np.sqrt(predStds)
    plt.plot(x[:,1], predMeans, 'b')
    plt.plot(x[:,1], predMeans + predStds, 'b--')
    plt.plot(x[:,1], predMeans - predStds, 'b--')
    plt.xticks([-1, 0, 1])
    plt.yticks([-0.5, 0, 0.5])
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel("x")
    plt.ylabel("y")
