
import numpy as np
import random
import sys

from lib import rawdata
from lib import dataset
from lib import metrics
from lib import viz
from lib import preprocess
from lib import util

SAMPLE_SIZE = 100000

rand = False

def predictFires():
    #create a new Data and make burn names those three instead of all. Pass all 3 fires
    new_data = rawdata.RawData.load(burnNames='untrain', dates='all')
    newDataSet = dataset.Dataset(new_data, dataset.Dataset.vulnerablePixels)
    pointLst = newDataSet.toList(newDataSet.points)
    pointLst = random.sample(pointLst, SAMPLE_SIZE) #SAMPLE_SIZE
    test = dataset.Dataset(new_data, pointLst)
    return test

def openDatasets():
    data = rawdata.RawData.load(burnNames='all', dates='all')
    masterDataSet = dataset.Dataset(data, dataset.Dataset.vulnerablePixels)
    # print("Built master dataset")
    # print(masterDataSet.points)
    ptList = masterDataSet.sample(sampleEvenly=False)
    # ptList = random.sample(ptList, 5000)
    trainPts, validatePts, testPts =  util.partition(ptList, ratios=[.8,.9])
    train = dataset.Dataset(data, trainPts)
    validate = dataset.Dataset(data, validatePts)
    test = dataset.Dataset(data, testPts)
    return train, validate, test

def openAndPredict(weightsFile):
    from lib import model

    test = predictFires()
    test.save('testOtherFire')
    mod = getModel(weightsFile)
    predictions = mod.predict(test)
    util.savePredictions(predictions)
    res = viz.visualizePredictions(test, predictions)
    viz.showPredictions(res)
    return test, predictions

def openAndTrain():
    from lib import model

    # train, validate, test = openDatasets()
    # OR test = dataset.openDataset(datasetfname)
    # train.save('train')
    # test.save('test')
    # validate.save('validate')

    path = 'output/datasets/'
    train = dataset.openDataset(path + 'train03Sep')
    validate = dataset.openDataset(path + 'validate03Sep')
    test = dataset.openDataset(path + 'test03Sep')


    mod, pp = getModel()

    model.fireCastFit(mod, pp, train, validate)
    predictions, _ = model.fireCastPredict(mod, pp, test, rand)
    calculatePerformance(test, predictions)

    return test, predictions


    # mod.fit(train, validate)
    # mod.saveWeights()
    # predictions, _ = mod.predict(test, rand)
    # util.savePredictions(predictions)
    # return test, predictions

def calculatePerformance(test, predictions):

    fireDate = []
    samples = []
    preResu = []

    print("SIZE OF PREDICTIONS: " , len(predictions))
    for pre in predictions:
        fireDate.append(pre[1])
        samples.append(pre[2])
        preResu.append(predictions.get(pre))

    # viz.compare_farsite(test, samples, preResu, len(predictions), fireDate)
    # viz.getNumbersNonConsecutive(test, samples, preResu, len(predictions), fireDate)
    viz.getNumbers(test, samples, preResu, len(predictions), fireDate)
    res = viz.visualizePredictions(test, predictions, preResu)
    viz.showPredictions(res)



def reloadPredictions():
    predFName = "09Nov10:39.csv"
    predictions = util.openPredictions('output/predictions/'+predFName)
    test = dataset.Dataset.open("output/datasets/test.json")
    return test, predictions

def getModel(weightsFile=None):
    from lib import model
    print('in getModel')
    numWeatherInputs = 8
    usedLayers = ['dem','ndvi', 'aspect', 'slope', 'band_2', 'band_3', 'band_4', 'band_5'] #, 'slope'
    AOIRadius = 30
    pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

    # mod = model.FireModel(pp, weightsFile)
    mod = model.fireCastModel(pp, weightsFile)
    return mod, pp

def example():
    try:
        modfname = sys.argv[1]
        datasetfname = sys.argv[2]
        print("working")
    except:
        print('about to import tkinter')
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        print('done!')
        root = Tk()
        print('Tked')
        root.withdraw()
        print('withdrawn')
        modfname = askopenfilename(initialdir = "models/",title="choose a model")
        datasetfname = askopenfilename(initialdir = "output/datasets",title="choose a dataset")
        root.destroy()



    test = dataset.openDataset(datasetfname)
    mod = getModel(modfname)
    predictions, resu = mod.predict(test, rand) #Could have messed this up by returning two things!!!!!!!!!
    # count = 0
    fireDate = []
    samples = []
    preResu = []



    print("SIZE OF PREDICTIONS: " , len(predictions))
    for pre in predictions:
        fireDate.append(pre[1])
        samples.append(pre[2])
        preResu.append(predictions.get(pre))
        # print(predictions.get(pre))
        # exit()
        # count = count + 1

    if rand:
        print("THIS IS A RANDOM TEST!!!")

    # for pt, pred in predictions.items():

    # viz.compare_farsite(test, samples, preResu, len(predictions), fireDate)
    # viz.getNumbersNonConsecutive(test, samples, preResu, len(predictions), fireDate)
    # viz.getNumbers(test, samples, preResu, len(predictions), fireDate)
    res = viz.visualizePredictions(test, predictions, preResu)
    viz.showPredictions(res)

#uncomment openAndTrain() to train a new model
#if you create a new dataset, must change name to have "_" instead of "/" for it to work with example()
# openAndTrain()

#uncomment the next two lines to create a validation dataset
# test = predictFires()
# dataset.Dataset.save(test)

# openAndPredict('') #enter weightsFile

#uncomment example() to make test images appear
#To run: python3 main.py models/model_name output/datasets/dataset_name
#final model: 11Apr17_55.h5
#test dataset: 11Apr20_49.              #this has 500,000 points

# example()

if len(sys.argv) == 3:
    example()
elif len(sys.argv) == 2:
    print('Please include a model and a dataset.')
    exit()
elif len(sys.argv) == 1:
    print('Training a new model...')
    openAndTrain()
else:
    test = predictFires()
    dataset.Dataset.save(test, fname='beaverCreek')
