import json
from extractorIndices import *
from Individuo import *
from Poblacion import *
from genetic import *

#------------------------funciones de ejecucion de programa-----------------------------------    
def PerformExperiment(Parametros, dataframe, dataTest):
    """
        Executes the experiments with different parameters and saves the result of each one in a file

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

        Parameters 
        ----------
        Parametros: List of experiments with the information of each one
        dataframe: Data where to execute the genetic algorithm (Train)
        dataTest: Data where to test (Not-seen data) the genetic resuñt

        Returns 
        ----------
        Results of experiments
    """
    return [PerformExperiment(experimentParameter, data, dataTest) for experimentParameter in Parametros]
 

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
    results = []
    if ModoDeEjecucionTest:
        #this experiments are performed just to compare different parameters applied on the same initial poblation
        #we could modify it just to run different algorithms (different initial poblations) easily
        results = PerformSetOfExperiments(Parametros, dfTrain , dfTest)
    else:
        #run with the set of parameters decided
        results = [PerformExperiment(Parametros[0], dfTrain , dfTest)]
    #save it into a file to trace
    
    #show a plot for documentation and graphical view
    #plotResults(results, Parametros)
    