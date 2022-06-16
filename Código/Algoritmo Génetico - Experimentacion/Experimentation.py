import requests
import numpy as np
import random as rng
import sys
import matplotlib.pyplot as plt
import json
import math
from extractorIndices import *
from Individuo import *
from Poblacion import *
from genetic import *
#adjust for passing parameters and return a value in multithreading
from threading import Thread
import multiprocessing

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)

#------------------------funciones de ejecucion de programa-----------------------------------    
def PerformExperiment(Parametros, dataframe, dataTest):
    """
        Executes the experiments with different parameters and saves the result of each one in a file
        It executes each experiment in each threads

        Parameters 
        ----------
        Parametros: List of experiments with the information of each one
        dataframe: Data where to execute the genetic algorithm (Train)
        dataTest: Data where to test (Not-seen data) the genetic resuñt

        Returns 
        ----------
        Results of all experiments
    """
    # Iniciar algoritmo
    print(f"Inicializando {Parametros['Experimento']} poblacion inicial...")
    algortimoGenetico = AlgritmoGenetico(Parametros["NumeroIndividuos"], dataframe, Parametros["initialBudget"], Parametros["pricePerOperation"],  Parametros["tournamentSizeRate"], Parametros["PMutacion"], Parametros["Cruce"], Parametros["ProbSwitchMulti"], Parametros["IndicadoresPorIndividuo"], Parametros["sizeBitsIndicator"])
    print(f"Inicializando algortimo {Parametros['Experimento']}...")
    result = algortimoGenetico.run(Parametros["NumeroIteraciones"])
    print(f"Algoritmo  {Parametros['Experimento']} terminado!")
    result['BestIndividuoObj'].resetFit(dataTest)
    test = result['BestIndividuoObj'].computeFitness()
    result["testResults"] = test
    print(f"Resultados Test: {test}")
    saveResults(result)
    return result


def PerformSetOfExperiments(Parametros, data, dataTest):
    """
        Executes the experiment with its parameters 
        It executes each experiment in each threads

        Parameters 
        ----------
        Parametros: List of experiments with the information of each one
        dataframe: Data where to execute the genetic algorithm (Train)
        dataTest: Data where to test (Not-seen data) the genetic resuñt

        Returns 
        ----------
        Results of experiments
    """
    #we will run simultaneously some algorithms to save time
    threadsFree = (multiprocessing.cpu_count()-1)
    experimentsToRun = min(threadsFree, len(Parametros))
    Experiments = []
    #Indicate the experiments we will run
    for experimentIndex in range(experimentsToRun):
        experimento = ThreadWithReturnValue(target = PerformExperiment ,args=[Parametros[experimentIndex], data, dataTest])
        Experiments.append(experimento)
    #Start the experiments
    for experiment in Experiments:
        experiment.start()
    #Wait until they finish
    for experiment in Experiments:
        experiment.join()

def loadParameters():
    """
        Open params file
    """
    with open("params.json", "r+") as file:
        ParametrosFile = json.load(file)
        return ParametrosFile["Experimentos"]

def splitContinous(trainingRatio, df):
    """
        Splits a data set into train and test data in such a way that the last part is for test

        Parameters 
        ----------
        trainingRatio: Percentage of data for trainingInstances (From instance 0 to TotalInstances*ratio)
        dataframe: Data where to execute the genetic algorithm (Train)

        Returns 
        ----------
        Results the two seaprated dataFrames
    """
    dfInstances = len(df.index)
    trainingInstances = round(dfInstances * trainingRatio)
    dfTrain = df.iloc[:trainingInstances,:]
    dfTest = df.iloc[trainingInstances:,:]
    return [dfTrain, dfTest]

def saveResults(result):
    """
        Takes result from one experiment and stores it  in output file

        Parameters 
        ----------
        result: Result from one experiment
        
    """
    with open("OutputResults.json", "r+") as file:
        data = json.load(file)
        del result["BestIndividuoObj"]
        data["Results"].append(result)
        file.seek(0)
        json.dump(data, file)

if __name__ == '__main__':
    filename = "./dataToTrainEvaluation/F-2022-05-19.pkl"
    df = pd.read_pickle(filename)
    Parametros = loadParameters()
    trainingRatio = 0.7
    dfTrain , dfTest = splitContinous(trainingRatio, df)
    #check if we wanna compare experiments or just run the final solution
    ModoDeEjecucionTest = True
    #check if we are in test zone
    if ModoDeEjecucionTest:
        #this experiments are performed just to compare different parameters applied on the same initial poblation
        #we could modify it just to run different algorithms (different initial poblations) easily
        PerformSetOfExperiments(Parametros, dfTrain , dfTest)
    else:
        #run with the set of parameters decided
        PerformExperiment(Parametros[0], dfTrain , dfTest)
    #save it into a file to trace
    
    #show a plot for documentation and graphical view
    #plotResults(results, Parametros)
    