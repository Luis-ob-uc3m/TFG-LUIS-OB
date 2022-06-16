import random as rng
import numpy as np
from extractorIndices import *
from Individuo import *
import matplotlib.pyplot as plt
import math

class Poblacion:
    def __init__(self, numeroDeIndividuos, dataFrame, initialBudget, pricePerOperation, numeroDeIndicadoresPorIndividuo=3, sizeBitsIndicator=4, referencePrice = 'close'):
        """
        Initiates the object, as a set of indivuals based on parameters
        
        Parameters (Optional)
        ----------
        numeroDeIndividuos: Number of individuals in the poblation
        initialBudget: Budget of each individual to operate with,
        pricePerOperation: Amount of money to exped each time we buy/sell
        numberOfIndicators: Numbers of Technical Indicators used in the codification (Default = 3), 
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification (Default = 4), 
        dataFrame: Temporal data used for choosing actions,
        referencePrice: Label to consider currentPrice (Default = 'close')
        """
        self.sizePoblacion = numeroDeIndividuos
        self.numeroDeIndicadoresPorIndividuo = numeroDeIndicadoresPorIndividuo
        self.sizeBitsIndicator = sizeBitsIndicator
        self.data = dataFrame
        self.pricePerOperation = pricePerOperation
        self.initialBudget = initialBudget
        self.referencePrice = referencePrice 
        self.individuos = self.generarPoblacionRandom(numeroDeIndicadoresPorIndividuo, sizeBitsIndicator)
    
    def computePoblationFitness(self):
        """
        Computes the fit of each individual    
        
        Returns 
        ----------
        Best fit of the poblation
        """
        bestFit = 0
        for individuo in self.individuos:
            fit = individuo.computeFitness()
            if fit > bestFit: bestFit = fit
        return bestFit
        
    def generarPoblacionRandom(self, numeroDeIndicadoresPorIndividuo, sizeBitsIndicator):
        """
        Generates the individual objects with random valid chromosomes

        Parameters 
        ----------
        numberOfIndicators: Numbers of Technical Indicators used in the codification
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification 

        Returns 
        ----------
        Returns new set of individuals as an array
        """
        return [Individuo(numberOfIndicators = numeroDeIndicadoresPorIndividuo, sizeBitsIndicator =sizeBitsIndicator, data = self.data, initialBudget = self.initialBudget, referencePrice = self.referencePrice, pricePerOperation = self.pricePerOperation) for individuo in range(self.sizePoblacion)]
    
    def getBestIndividuo(self):
        """
        Sorts the individuals by descending fit and returns the first one

        Returns 
        ----------
        Returns best individual object
        """
        qualy = sorted(self.individuos, key= lambda x: float(x.fitness), reverse=True)
        return qualy[0]
    
    def getFitnessAllIndividuos(self):
        """
        Sorts the individuals by descending fit and returns all fits

        Returns 
        ----------
        Returns all fits
        """
        qualy = sorted(self.individuos, key= lambda x: float(x.fitness), reverse=True)
        return [individual.fitness for individual in qualy]
    
    def getWorstIndividuo(self):
        """
        Sorts the individuals by descending fit and returns the last one

        Returns 
        ----------
        Returns worst individual object
        """
        qualy = sorted(self.individuos, key= lambda x: float(x.fitness), reverse=True)
        return qualy[-1]

    def getBestIndividuoValue(self):
        """
        Sorts the individuals by descending fit and returns the first one

        Returns 
        ----------
        Returns best individual fitness
        """
        return self.getBestIndividuo().fitness

    def getWorstIndividuoValue(self):
        """
        Sorts the individuals by descending fit and returns the last one

        Returns 
        ----------
        Returns worst individual fitness
        """
        return self.getWorstIndividuo().fitness

    def getInfoIndividuos(self):
        """
        Gets info from all individuals

        Returns 
        ----------
        Returns the individuals as an array
        """
        infoIndividuos = map(lambda ind: str(ind), self.individuos)
        return '\n'.join(infoIndividuos)

    def getDetailedInfoIndividuos(self):
        """
        Gets info from all individuals

        Returns 
        ----------
        Returns the individuals info in a dict
        """
        infoIndividuos =  map(lambda ind: ind.getInfo(), self.individuos)
        return '\n'.join(infoIndividuos)
    
    def getMeanFit(self):
        """
        Computes the mean of fitness values over the poblation

        Returns 
        ----------
        Returns the mean fit
        """
        fmean = 0
        for individuo in self.individuos:
            fmean += float(individuo.fitness)
        return (fmean / self.sizePoblacion)
    
    def getFr(self, individuoFR):
        """
        Computes the relative frequency of an individual over the poblation

        Parameters 
        ----------
        individuoFR: Individual object to compute value

        Returns 
        ----------
        Returns the relative frequency
        """
        ftotal = 0
        for individuo in self.individuos:
            ftotal += float(individuo.fitness)
        return (float(individuoFR.fitness) / ftotal)
    #Desviacion
    def getDiversity(self):
        """
        Computes the diversity of the poblation

        Returns 
        ----------
        Returns the diversity value (Desviation)
        """
        sumatory = 0
        fmean = self.getMeanFit()
        for ind in self.individuos:
            frelative = self.getFr(ind)
            sumatory += ((frelative - fmean)**2)
        return math.sqrt(sumatory)

    def getPS(self):
        """
        Computes the selective pressure over the poblation

        Returns 
        ----------
        Returns the selective pressure 
        """
        fmax = float(self.getBestIndividuoValue())
        fmean = self.getMeanFit()
        return fmax / fmean #if were maximization
        # if fmax != 0:
        #     return  fmean / fmax # as minimization
        # else:
        #     return -1
    def __str__(self):
        """
        Computes the pair chromosome|fitness for each individual

        Returns 
        ----------
        Returns the poblation information 
        """
        return '\n'.join(" , ".join([str(individuo), str(individuo.fitness)]) for individuo in self.individuos)

if __name__ == '__main__':
    filename = "./dataTry.pkl"
    df = pd.read_pickle(filename)
    nInds = 100
    poblacion = Poblacion(nInds, df)
    print(poblacion)
    for individuo in poblacion.individuos:
        print(individuo.getInfo())