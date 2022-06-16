import requests
import numpy as np
import random as rng
import sys
import matplotlib.pyplot as plt
import pandas as pd
from extractorIndices import *
from Individuo import *
from Poblacion import *
#adjust for passing parameters and return a value in multithreading


class AlgritmoGenetico:
    def __init__(self, numeroIndividuosPoblacion, dataframe, initialBudget = 10000, pricePerOperation=1000 , tournamentSizeRate=0.05, ProbMutacion=0.1, TipoCruce="CruceUniforme", probSwitchMultipunto=0.025, numIndicadoresPorIndividuo = 3, sizeBitsIndicator=4):
        """
        Initiates the object, with a poblation of rnadom individuals and essential information
        
        Parameters (Optional)
        ----------
        numeroDeIndividuos: Number of individuals in the poblation
        initialBudget: Budget of each individual to operate with,
        pricePerOperation: Amount of money to exped each time we buy/sell
        numberOfIndicators: Numbers of Technical Indicators used in the codification (Default = 3), 
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification (Default = 4), 
        dataFrame: Temporal data used for choosing actions,
        referencePrice: Label to consider currentPrice (Default = 'close')
        tournamentSizeRate: Percentage of individuals to go to tournament (Default = 0.05)
        ProbMutacion: Value to compute randomly when to mutate a bit (Default = 0.1)
        TipoCruce: Select the reproduction way (Default = CruceUniforme)
        probSwitchMultipunto: Value to compute randomly when to cut chromosome (Higher value, more cuts. Default = 0.025)
        """
        self.numIndividuos = numeroIndividuosPoblacion
        self.ProbMutacion = float(ProbMutacion)
        self.TipoCruce = TipoCruce
        self.data = dataframe
        self.initialBudget = initialBudget
        self.pricePerOperation = pricePerOperation
        # A minimum of 1 individual should be selected to win the tournament, will always win
        self.sizeTournament = max(int(numeroIndividuosPoblacion * tournamentSizeRate), 1)
        self.poblacion = Poblacion(numeroIndividuosPoblacion, dataframe, initialBudget,pricePerOperation, numIndicadoresPorIndividuo, sizeBitsIndicator)
        self.probSwitchMultipunto = probSwitchMultipunto

    def getBestNIndividuos(self, individuosList, podiumSize=1):
        """
        Gets the best n individuals from an individual list    
        Parameters 
        ----------
        individuosList: Individuals poblation to compute, 
        podiumSize: Size of individuals to get (Default = 1), 
        
        Returns 
        ----------
        list of best n individuals
        """
        qualy = sorted(individuosList, key=lambda x: float(x.fitness), reverse=True)
        if (podiumSize == 1):
            return qualy[0]
        return qualy[:podiumSize]

    def evaluacion(self, DataHistorical, iteration):
        """
        Computes the fitness and evaluates the individuals  
        Parameters 
        ----------
        DataHistorical: dict to save information, 
        iteration: current iteration, 
        
        Returns 
        ----------
        Information stored in dict
        """
        # Evaluacion
        self.poblacion.computePoblationFitness()
        bestIndividuo = self.poblacion.getBestIndividuo()
        # Guardar Datos
        if (iteration == 0 or float(self.poblacion.getBestIndividuoValue()) > float(DataHistorical["BestFitness"])):
            DataHistorical["BestFitness"] = (self.poblacion.getBestIndividuoValue())
            DataHistorical["BestIndividuo"] = str(bestIndividuo)
            DataHistorical["BestIndividuoObj"]  = bestIndividuo
    
        
           

        DataHistorical["ResultadoObtenido"].append(
            {
                "CurrentIteration": iteration,
                "BestFitnessOfthisIteration": self.poblacion.getBestIndividuoValue(),
                "BestIndividualOfthisIteration": str(bestIndividuo),
                "WorstFitnessOfthisIteration": self.poblacion.getWorstIndividuoValue(),
                "WorstIndividualOfthisIteration": str(self.poblacion.getWorstIndividuo()),
                "PresionSelectiva": self.poblacion.getPS(),
                "Diversidad": self.poblacion.getDiversity(),
                'BestIndividuoInfoExtra' : {
                    'NumberOfOpertations':bestIndividuo.operationsPerformed,
                    'NumberOfBuys':bestIndividuo.numberofbuys,
                    'NumberOfSells':bestIndividuo.numberofsells,
                    'HistoricalOperations': bestIndividuo.numberOfPositiveOperations,
                    'HistoricalComparisions': bestIndividuo.historicalComparision,
                    "BeneficioMedioPorOperacion":bestIndividuo.fitness/bestIndividuo.numberOfPositiveOperations,
                    "GananciaMaxima":bestIndividuo.gananciaMaxima,
                    "GananciaMaximaPorOperación":bestIndividuo.gananciaMaxima/bestIndividuo.numberOfPositiveOperations,
                    "PerdidaMaxima":bestIndividuo.perdidaMaxima,
                    "PerdidaMaximaPorOperación":bestIndividuo.perdidaMaxima/bestIndividuo.numberOfPositiveOperations,
                    "PositiveOperationsRatio":bestIndividuo.numberOfPositiveOperations /bestIndividuo.numberOfPositiveOperations,
                },
                "Fits": self.poblacion.getFitnessAllIndividuos()
            }
        )
        return DataHistorical

    def seleccion(self):
        """
        Applies tournament selection to get best individuals and mantain diversity 
        
        Returns 
        ----------
        New poblation from tournament results
        """
        newPoblacion = []
        for tournamentRound in range(self.poblacion.sizePoblacion):
            # take a random subset of individuals to compete to be the best
            tournamentBranch = rng.sample(self.poblacion.individuos, self.sizeTournament)
            best = self.getBestNIndividuos(tournamentBranch)
            newPoblacion.append(best)
        return newPoblacion

    # Cruce recibe los individuos padres, retorna una lista ya de individuos como objetos totalmente diferente para no tener que modificar a 1 padres y eliminar al otro --> mas sencillo
    def crucePoblacion(self, individuosAReproducir, tipoDecruce, probSwitchMultipunto=0.025):
        """
        Creates new poblation by reproducing current individuals 

        Parameters 
        ----------
        individuosAReproducir: Individuals poblation to reproduce, 
        tipoDecruce: Type of reprodution, 
        probSwitchMultipunto: In case type = "CruceMultipunto"

        Returns 
        ----------
        New poblation
        """
        newIndPoblacion = []
        # Para que la población hija sea igual de tamaño , elegimos 2 progenitores las mismas veces --> tiene que haber parejas diferentes con mismos pades
        for individuo in individuosAReproducir:
            progenitor1 = rng.choice(individuosAReproducir)
            progenitor2 = rng.choice(individuosAReproducir)
            if(tipoDecruce == "CruceUniforme"):
                individuoHijo = self.cruceUniforme(progenitor1, progenitor2)
                newIndPoblacion.append(individuoHijo)
            elif(tipoDecruce == "CruceSimple"):
                hijos = self.cruceSimple(progenitor1, progenitor2)
                newIndPoblacion.append(hijos[0])
                newIndPoblacion.append(hijos[1])
            elif(tipoDecruce == "CruceMultipunto"):
                hijos = self.cruceMultipunto(progenitor1, progenitor2, probSwitchMultipunto)
                newIndPoblacion.append(hijos[0])
                newIndPoblacion.append(hijos[1])
            else:
                print("You must select a valid method for cross individuals")
                sys.exit(0)
        return newIndPoblacion

    def cruceUniforme(self, progenitor1, progenitor2):
        """
        Creates new individual by appliying uniform crossover

        Parameters 
        ----------
        progenitor1: Individual 1 to reproduce
        progenitor2: Individual 2 to reproduce

        Returns 
        ----------
        New Individual, outcome of crossover
        """
        adnhijo = []
        # Reproduccion de 2 individuos del mismo tamaño
        for adnIndex in range(progenitor1.chromosomeLen):
            adn1 = progenitor1.chromosomeAsArray[adnIndex]
            adn2 = progenitor2.chromosomeAsArray[adnIndex]
            adnhijo.append(rng.choice([adn1, adn2]))
        return  Individuo(adnhijo, self.poblacion.numeroDeIndicadoresPorIndividuo,self.poblacion.sizeBitsIndicator,  data = self.data, initialBudget = self.initialBudget, pricePerOperation = self.pricePerOperation)
        
    def cruceSimple(self, progenitor1, progenitor2):
        """
        Creates new individual by appliying simple crossover

        Parameters 
        ----------
        progenitor1: Individual 1 to reproduce
        progenitor2: Individual 2 to reproduce

        Returns 
        ----------
        New Individual, outcome of crossover
        """
        splitIndex = rng.randint(0, progenitor1.chromosomeLen)
        adnhijo1 = progenitor1.chromosomeAsArray[:splitIndex] + progenitor2.chromosomeAsArray[splitIndex:]
        adnhijo2 = progenitor2.chromosomeAsArray[:splitIndex] + progenitor1.chromosomeAsArray[splitIndex:]
        hijo1 = Individuo(adnhijo1, self.poblacion.numeroDeIndicadoresPorIndividuo,self.poblacion.sizeBitsIndicator, data = self.data, initialBudget = self.initialBudget, pricePerOperation = self.pricePerOperation)
        hijo2 = Individuo(adnhijo2, self.poblacion.numeroDeIndicadoresPorIndividuo,self.poblacion.sizeBitsIndicator,  data = self.data, initialBudget = self.initialBudget, pricePerOperation = self.pricePerOperation)
        return [hijo1, hijo2]

    def cruceMultipunto(self, progenitor1, progenitor2, probSwitch=0.025):
        """
        Creates new individual by appliying multipoint crossover

        Parameters 
        ----------
        progenitor1: Individual 1 to reproduce
        progenitor2: Individual 2 to reproduce
        probswitch: probability to generate a new cut

        Returns 
        ----------
        New Individual, outcome of crossover
        """
        adnhijo1 = []
        adnhijo2 = []
        chromosomeProg1 = []
        chromosomeProg2 = []
        # make a copy to avoid reference problems
        for adn in progenitor1.chromosomeAsArray:
            chromosomeProg1.append(adn)
        for adn in progenitor2.chromosomeAsArray:
            chromosomeProg2.append(adn)
        # Reproduccion de 2 individuos del mismo tamaño
        for adnIndex in range(progenitor1.chromosomeLen):
            if(rng.random() <= probSwitch):
                chromosomeProg1, chromosomeProg2 = chromosomeProg2, chromosomeProg1
            adn1 = chromosomeProg1[adnIndex]
            adn2 = chromosomeProg2[adnIndex]
            adnhijo1.append(adn1)
            adnhijo2.append(adn2)
        hijo1 = Individuo(adnhijo1, self.poblacion.numeroDeIndicadoresPorIndividuo,self.poblacion.sizeBitsIndicator, data = self.data, initialBudget = self.initialBudget, pricePerOperation = self.pricePerOperation)
        hijo2 = Individuo(adnhijo2, self.poblacion.numeroDeIndicadoresPorIndividuo,self.poblacion.sizeBitsIndicator,  data = self.data, initialBudget = self.initialBudget, pricePerOperation = self.pricePerOperation)
        return [hijo1, hijo2]

    # Cruce recibe los ya de individuos como objetos por lo que no hace falta crearlos de nuevo
    def mutacionPoblacion(self, individuosAMutar=None, probabilidadMutacion=None):
        """
        Mutates each individual from a poblation

        Parameters 
        ----------
        individuosAMutar: Individual list to mutate
        probabilidadMutacion: Probability to switch a bit

        Returns 
        ----------
        New poblation of individuals
        """
        if probabilidadMutacion is None: probabilidadMutacion = self.ProbMutacion
        if individuosAMutar is None: individuosAMutar = self.poblacion.individuos

        newIndPoblacion = []
        for individuo in individuosAMutar:
            # modifica la individuo al pasar por "referencia" (assignment) en python
            individuo = self.mutacionIndividuo(individuo, probabilidadMutacion)
            newIndPoblacion.append(individuo)
        self.poblacion.individuos = newIndPoblacion
        return newIndPoblacion

    def mutacionIndividuo(self, individuo, probabilidadMutacion=None):
        """
        Mutates an individual 

        Parameters 
        ----------
        individuo: Individual  to mutate
        probabilidadMutacion: Probability to switch a bit

        Returns 
        ----------
        New individual
        """
        if probabilidadMutacion is None: probabilidadMutacion = self.ProbMutacion
        for adnIndex in range(individuo.chromosomeLen):
            if rng.random() <= probabilidadMutacion:
                individuo.chromosomeAsArray[adnIndex] = int(
                    not individuo.chromosomeAsArray[adnIndex])
        return Individuo(individuo.chromosomeAsArray, self.poblacion.numeroDeIndicadoresPorIndividuo,self.poblacion.sizeBitsIndicator,  data = self.data, initialBudget = self.initialBudget, pricePerOperation = self.pricePerOperation)

    def run(self, nIterations=1):
        """
        Executes genetic algorithm

        Parameters 
        ----------
        nIterations: Maximal iterations to execute 

        Returns 
        ----------
        Historical data from execution
        """
        DataHistorical = {
            "nIterations": nIterations,
            "numIndividuos": self.numIndividuos,
            "nEvaluations": nIterations * self.numIndividuos,
            "ProbMutacion": self.ProbMutacion,
            "sizeTournament": self.sizeTournament,
            "TipoCruce": self.TipoCruce,
            "BestIndividuo": None,
            "BestIndividuoObj":None,
            "BestFitness": None,
            'BestIndividuoInfoExtra' : {
                'NumberOfOpertations':0,
                'NumberOfBuys':0,
                'NumberOfSells':0,
                'HistoricalOperations': [],
                'HistoricalComparisions': [],
                "BeneficioMedioPorOperacion":None,
                "GananciaMaxima":None,
                "GananciaMaximaPorOperación":None,
                "PerdidaMaxima":None,
                "PerdidaMaximaPorOperación":None,
                "PositiveOperationsRatio":None,
            },
            
            "ResultadoObtenido": []
        }
        numGenSinMejora = 0
        numGenSeguidasSinMejora = 0
        valorActualPoblacion = None
        valorAnteriorPoblacion = None
        iteration = 0
        minimiNumGenSinMejora = 25
        MinimumCheck = numGenSinMejora<minimiNumGenSinMejora
        #condicion de parada --> numero de generaciones sin superar un sumbral de mejora
        # numero maximo de generaciones para limite 
        while(MinimumCheck and iteration < nIterations):
            print(f"Iteration {iteration}...")
            # Las evaluaciones se realizan al iniciar cada individuo de la poblacion.
            # Es decir, en el procedimiento seguido, 1ero al inicializar el agoritmo y despues de cada mutacion (último operador de la generacion)
            # Es por ello que numero de evaluaciones es nIterations + 1
            self.evaluacion(DataHistorical, iteration)
            
            #check parada
            if(iteration>0):
                valorAnteriorPoblacion = valorActualPoblacion
            valorActualPoblacion= float(self.poblacion.getBestIndividuoValue())
            if(iteration >0 and ((float(DataHistorical["BestFitness"]) - valorActualPoblacion) > 0)): #CASE NOT GETTING BETTER
                numGenSinMejora +=1
                numGenSeguidasSinMejora +=1
            elif(iteration >0 and ((float(DataHistorical["BestFitness"]) - float(valorActualPoblacion)) == 0)): #CASE SAME, CHECK TENDENCY WITH LAST VALUE
                if( (valorActualPoblacion - valorAnteriorPoblacion > 0)): #IF GOING UP RESET
                    numGenSeguidasSinMejora =0
                else:#IF GOING UP 
                    numGenSinMejora +=1
                    numGenSeguidasSinMejora +=1
            else: #CASE WE GOT BETTER
                numGenSeguidasSinMejora = 0
            
            if numGenSinMejora>=minimiNumGenSinMejora:
                MinimumCheck = numGenSeguidasSinMejora < 5
            

            individuosAReproducir = self.seleccion()
            nuevosHijos = self.crucePoblacion(individuosAReproducir, self.TipoCruce, self.probSwitchMultipunto)
            nuevosIndividuos = self.mutacionPoblacion(nuevosHijos, self.ProbMutacion)
            self.poblacion.individuos = nuevosIndividuos
            iteration +=1
            if(iteration%10==0):
                print ("----------------------------New checkpoint!-----------------------------------")
                print (f"Currently iteration is {iteration}")
                print (f"Best  fitness value is {DataHistorical['BestFitness']}")
                print (f"The codification is: {DataHistorical['BestIndividuo']}")
                print (f"Evaluations: {iteration * DataHistorical['numIndividuos']}")
                print ("----------------------------Keep Working....------------------------------")

        
        DataHistorical["nIterations"] = len(DataHistorical["ResultadoObtenido"])
        DataHistorical["nEvaluations"] = DataHistorical["nIterations"] * DataHistorical["numIndividuos"]
        DataHistorical["BestIndividuoInfoExtra"]["NumberOfOpertations"] = DataHistorical['BestIndividuoObj'].operationsPerformed
        DataHistorical["BestIndividuoInfoExtra"]["NumberOfBuys"] = DataHistorical['BestIndividuoObj'].numberofbuys
        DataHistorical["BestIndividuoInfoExtra"]["NumberOfSells"] = DataHistorical['BestIndividuoObj'].numberofsells
        DataHistorical["BestIndividuoInfoExtra"]["HistoricalOperations"] = DataHistorical['BestIndividuoObj'].historicalOperations
        DataHistorical["BestIndividuoInfoExtra"]["HistoricalComparisions"] = DataHistorical['BestIndividuoObj'].historicalComparision

        DataHistorical["BestIndividuoInfoExtra"]["BeneficioMedioPorOperacion"] = DataHistorical['BestFitness'] / DataHistorical["BestIndividuoInfoExtra"]["NumberOfOpertations"] 
        DataHistorical["BestIndividuoInfoExtra"]["GananciaMaxima"] = DataHistorical['BestIndividuoObj'].gananciaMaxima
        DataHistorical["BestIndividuoInfoExtra"]["GananciaMaximaPorOperación"] = DataHistorical["BestIndividuoInfoExtra"]["GananciaMaxima"]  / DataHistorical["BestIndividuoInfoExtra"]["NumberOfOpertations"] 
        DataHistorical["BestIndividuoInfoExtra"]["PerdidaMaxima"] = DataHistorical['BestIndividuoObj'].perdidaMaxima
        DataHistorical["BestIndividuoInfoExtra"]["PerdidaMaximaPorOperación"] =  DataHistorical["BestIndividuoInfoExtra"]["PerdidaMaxima"] / DataHistorical["BestIndividuoInfoExtra"]["NumberOfOpertations"] 
        DataHistorical["BestIndividuoInfoExtra"]["PositiveOperationsRatio"] = DataHistorical['BestIndividuoObj'].numberOfPositiveOperations / DataHistorical["BestIndividuoInfoExtra"]["NumberOfOpertations"] 

        print({"Evaluaciones": DataHistorical['nEvaluations'] ,"Value":DataHistorical["BestFitness"], "Individuo":DataHistorical['BestIndividuo']})
        return DataHistorical
        # return {"IndividuoChromosome":MinIndividuo.getChromosome(), "IndividuoValue":MinValueGet}



if __name__ == '__main__':
    filename = "./dataTry.pkl"
    #df = pd.read_pickle(filename)
    algortimoGenetico = AlgritmoGenetico(10,None,10000,1000, 0.3, 0.1, "CruceMultipunto", 0.063, 3, 4)
    chromosome = '11111111111111111'
    myInd = Individuo(chromosome, numberOfIndicators = 3, sizeBitsIndicator = 4, fitness = None, data = None, initialBudget=0, pricePerOperation=None,  referencePrice='close', autoComputeFit = False)
    newInd = algortimoGenetico.mutacionIndividuo(myInd,1)
    #result = algortimoGenetico.run(10)
    print(newInd)

    

























