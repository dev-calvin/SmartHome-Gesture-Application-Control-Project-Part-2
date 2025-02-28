# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import math
import pandas as pd

from frameextractor import frameExtractor

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor

def main():
    testPaths, testFrames = getAllMiddleFrames('test')
    trainPaths, trainFrames = getAllMiddleFrames('traindata')

    testFeatureVectors = extractFeatureVectors('test', testFrames)
    trainFeatureVectors = extractFeatureVectors('traindata', trainFrames)

    results = os.listdir('/')

    results = []
    for index, test in enumerate(testPaths):
        results.append(findClosest(testFeatureVectors[index], trainFeatureVectors))

    # trainLabels = []
    # for i in range(len(results)):
    #     trainFile = trainPaths[results[i]].split('\\')[-1]
    #     trainLabel = trainFile.split('.')[0]
    #     trainLabels.append(trainLabel)

    df = pd.DataFrame(results)
    df.to_csv('/Results.csv', header=False, index=False)

    return


def findClosest(test, trainFeatureVectors):
    closestIndex = 0
    cosSim = 0
    for index, vector in enumerate(trainFeatureVectors):
        if (getCosSim(test, vector) > cosSim):
            cosSim = getCosSim(test, vector)
            closestIndex = index
    return closestIndex


def getCosSim(test, train):
    cosSim = 0
    xx = 0
    xy = 0
    yy = 0
    for i in range(len(test)):
        x = test[i]
        y = train[i]
        xx += x*x
        xy += x*y
        yy += y*y
    cosSim = xy/math.sqrt(xx*yy)
    return cosSim


def extractFeatureVectors(type, frames):
    featureVectors = []
    for frame in range(frames):
        f = frame + 1
        featureVectors.append(extractHandShape(f'.\{type}Frames\{f:05d}.png')[0])

    return featureVectors


def getAllMiddleFrames(type):
    filePaths = getAllFilePaths(f'.\{type}')
    i = 0
    for path in filePaths:
        # frames.append(getMiddleFrame(path))
        frameExtractor(path,f'.\{type}Frames',i)
        i += 1
    return (filePaths, i)


def getAllFilePaths(directory):
    filePaths = []
    if os.path.exists(directory):
        for file in os.listdir(directory):
          filePath = os.path.join(directory, file)
          if os.path.isfile(filePath):
            filePaths.append(filePath)
    return filePaths


def extractHandShape(framePath):
    image = cv2.imread(framePath)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    featureExtractor = HandShapeFeatureExtractor.get_instance()
    extraction = featureExtractor.extract_feature(grayImage)
    return extraction

main()
