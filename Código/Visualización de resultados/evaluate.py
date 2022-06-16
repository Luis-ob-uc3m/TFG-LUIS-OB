import pandas as pd
#Move this file to Genetic to run, remove comments:
#from genetic import * 
#from Individuo import *
import json 

def saveResults(result, FILENAME="datesResults.json"):
    """
        Takes result from one experiment and stores it  in output file

        Parameters 
        ----------
        result: Result from one experiment
        
    """
    with open(FILENAME, "r+") as file:
        data = json.load(file)
        data["Results"].append(result)
        file.seek(0)
        json.dump(data, file)

def splitContinous(trainingRatio, df):
    dfInstances = len(df.index)
    trainingInstances = round(dfInstances * trainingRatio)
    dfTrain = df.iloc[:trainingInstances,:]
    dfTest = df.iloc[trainingInstances:,:]
    return [dfTrain, dfTest]

def trainGenetic(Parametros, dftrain):
    print("training...")
    algortimoGenetico = AlgritmoGenetico(Parametros["NumeroIndividuos"], dftrain, Parametros["initialBudget"], Parametros["pricePerOperation"], Parametros["tournamentSizeRate"], Parametros["PMutacion"], Parametros["Cruce"], Parametros["ProbSwitchMulti"], Parametros["IndicadoresPorIndividuo"], Parametros["sizeBitsIndicator"])
    geneticBot = algortimoGenetico.run(Parametros["NumeroIteraciones"])["BestIndividuoObj"]
    print(geneticBot.getInfo())
    return geneticBot

def testBot(geneticBot, dfTest):
    geneticBot.computeFitness(dfTest)
    print("test...")
    
if __name__ == '__main__':
    
    # Parametros =  {   
    #         "Experimento": 0,
    #         "Descripcion": "100",
    #         "NumeroIndividuos" : 100,
    #         "PMutacion" : 0.01,
    #         "tournamentSizeRate" : 0.225,
    #         "Cruce" : "CruceMultipunto",
    #         "ProbSwitchMulti": 0.063,
    #         "NumeroIteraciones" : 100,
    #         "IndicadoresPorIndividuo": 3,
    #         "sizeBitsIndicator":4,
    #         "initialBudget":1000
    #     }
    # geneticBot = trainGenetic(Parametros, dfTrain)
    # testBot(geneticBot, dfTest)
    individuoFord = '00001101011110001'
    individuoAAPL = '11100100011000001'
    individuoSAN = '01100001000100111'
    individuoBBVA = '00010000011000111'
    individuoMSFTTRAIN = '00101011100000011'
    individuoMSFTTEST = '00100111000100011'

    individuos = [[individuoFord,'Ford'], [individuoAAPL,'AAPL'],[individuoSAN,'SAN'],[individuoBBVA,"BBVA"], [individuoMSFTTRAIN, 'MSFTTRAIN'], [individuoMSFTTEST,'MSFTTEST']]
    
    
    filename = "./dataToTrainEvaluation/AAPL-2022-05-19.pkl"
    fileResults = "dataToTestEvaluation/dateResultsAAPL.json"
    individual = individuos[1]



    df = pd.read_pickle(filename)
    trainingRatio = 0.7
    dfTrain , dfTest = splitContinous(trainingRatio, df)
    if(len(individual[0])!=17):
        print("error in individual chromosome: ", individual[0])
    myInd = Individuo(individual[0],3,4,None,dfTrain,10000,1000)
    FIT = myInd.computeFitness(dfTrain,True)
    print(FIT)
    bestBudget = sorted(myInd.historicalOperations, key= lambda x: float(x['PotentialBudget']), reverse=True)
    previousOperationAction = "hold"
    previousOperationValue = 0
    positiveOperationsNumber = 0
    lastOperation = 0
    for operation in myInd.historicalOperations:
        if previousOperationAction == operation['Action']:print("WARNING! 2 CONSECUTIVES!!!!!")
        if ((operation['ActivesOperated'] * operation['CurrentPrice']) + previousOperationValue > 0):
            positiveOperationsNumber += 1
        previousOperationValue = operation['ActivesOperated'] * operation['CurrentPrice'] 
    if(myInd.stocksBuy + previousOperationValue > 0):
        positiveOperationsNumber += 1
    if(myInd.stocksBuy>0): lastOperation=1
    potentialMoney = myInd.stocksBuy * myInd.currentPrice
    if(myInd.currentBudget + potentialMoney > bestBudget[0]['PotentialBudget']):
        bestBudget[0]['PotentialBudget'] = myInd.currentBudget + potentialMoney
    Result = {
        "Name":individual[1], 
        "DataSet": 'dfTrain',
        "Noperations": myInd.operationsPerformed+lastOperation,
        "NBuys": myInd.numberofbuys,
        "NSells": myInd.numberofsells+lastOperation,
        "OperationsHistorical": myInd.historicalOperations,
        "Beneficio Maximo": bestBudget[0]['PotentialBudget'],
        "Beneficio Maximo Por Operacion": bestBudget[0]['PotentialBudget'] / myInd.operationsPerformed+lastOperation,
        "Perdida Maxima": bestBudget[-1]['PotentialBudget'],
        "Perdida Maxima Por Operacion": bestBudget[-1]['PotentialBudget'] / myInd.operationsPerformed+lastOperation,
        "PositiveOperations":positiveOperationsNumber,
        "Positive Operations Ratio":positiveOperationsNumber/(myInd.operationsPerformed+lastOperation)
    }
    saveResults(Result,fileResults)
    
    #----------------------------------------------------------------------------------------------------------------
    if(len(individual[0])!=17):
        print("error in individual chromosome: ", individual[0])
    myInd2 = Individuo(individual[0],3,4,None,dfTest,10000,1000)
    FIT = myInd2.computeFitness(dfTest,True)
    print(FIT)
    bestBudget = sorted(myInd2.historicalOperations, key= lambda x: float(x['PotentialBudget']), reverse=True)
    previousOperationAction = "hold"
    previousOperationValue = 0
    positiveOperationsNumber = 0
    lastOperation = 0
    for operation in myInd2.historicalOperations:
        if previousOperationAction == operation['Action']:print("WARNING! 2 CONSECUTIVES!!!!!")
        if ((operation['ActivesOperated'] * operation['CurrentPrice']) + previousOperationValue > 0):
            positiveOperationsNumber += 1
        previousOperationValue = operation['ActivesOperated'] * operation['CurrentPrice'] 
    if(myInd2.stocksBuy + previousOperationValue > 0):
        positiveOperationsNumber += 1
    if(myInd2.stocksBuy>0): lastOperation=1
    potentialMoney = myInd2.stocksBuy * myInd2.currentPrice
    if(myInd2.currentBudget + potentialMoney > bestBudget[0]['PotentialBudget']):
        bestBudget[0]['PotentialBudget'] = myInd2.currentBudget + potentialMoney
    Result = {
        "Name":individual[1], 
        "DataSet": 'dfTrain',
        "Noperations": myInd2.operationsPerformed+lastOperation,
        "NBuys": myInd2.numberofbuys,
        "NSells": myInd2.numberofsells+lastOperation,
        "OperationsHistorical": myInd2.historicalOperations,
        "Beneficio Maximo": bestBudget[0]['PotentialBudget'],
        "Beneficio Maximo Por Operacion": bestBudget[0]['PotentialBudget'] / myInd2.operationsPerformed+lastOperation,
        "Perdida Maxima": bestBudget[-1]['PotentialBudget'],
        "Perdida Maxima Por Operacion": bestBudget[-1]['PotentialBudget'] / myInd2.operationsPerformed+lastOperation,
        "PositiveOperations":positiveOperationsNumber,
        "Positive Operations Ratio":positiveOperationsNumber/(myInd2.operationsPerformed+lastOperation)
    }
    saveResults(Result,fileResults)
    #----------------------------------------------------------------------------------------------------------------







