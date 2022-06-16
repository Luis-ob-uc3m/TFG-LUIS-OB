import json
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
#Move this file to Genetic to run, remove comments:
#from Individuo  import *
#import extractorIndices 
import time

from time import sleep


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
def loadDataAsCSV(filename):
    csvFile = pd.read_csv(filename, index_col=0, parse_dates=True)
    csvFile.index = pd.to_datetime(csvFile.index)
    return csvFile

def plotData(dataCsv, columnName):
    dataCsv[columnName].plot(color='blue', label=f'{columnName} Price')
    plt.title('Extracción de datos del activo')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del activo')
    plt.legend()
    plt.show() 
    plt.clf()

def plotNext(dataCsv, columnName1, columnName2):
    figure, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].plot(dataCsv[columnName1], color='blue')
    axes[0].set_title('Open Price')
    axes[1].plot(dataCsv[columnName2], color='red')
    axes[1].set_title('Close Price')
    figure.tight_layout()
    plt.suptitle('Extracción de datos del activo')
    for axe in axes:
        axe.set_xlabel('Fecha')
        axe.set_ylabel('Valor del activo')
    plt.legend()
    plt.show() 
    plt.clf()

def plotCandleAsLine(dataCsv):
    plt.style.use('_mpl-gallery')
    fig, ax = plt.subplots()
    ax.fill_between(dataCsv.index, dataCsv['high'], dataCsv['low'], alpha=.5, linewidth=0)
    ax.plot(dataCsv['open'], color='green',  linestyle='dashed', label='Open Price')
    ax.plot(dataCsv['close'], color='red',  linestyle='dashed', label='Close Price')
    plt.title('Candle Data with lines')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del activo')
    plt.legend()
    plt.show() 
    plt.clf()

def candlestickChart(dataCsv):
    mpf.plot(dataCsv['2021-01':'2022-01'], type="candle", title="Stock price",
    mav=(20) ,volume=True, tight_layout=True, style="yahoo")
    
def getTimeElapsedOperations(df):
    lastDate = None
    lastAction = None
    rowCounter = 0
    Summary = []
    for date, row in df.T.iteritems():
        if rowCounter == 0:
            lastDate = date
            lastAction = row["Action"]
            rowCounter += 1
            continue
        Info = {
            "Action": row["Action"],
            "Date": row["Day"],
            "Last Action": lastAction,
            "Time Elapsed": date-lastDate
        }
        Summary.append(Info)
        lastDate = date
        rowCounter += 1
        lastAction = row["Action"]
    return Summary




def plotOperations(dataCsv, columnName, operation):
    dataCsv[columnName] =  dataCsv[columnName] / 25
    ax = dataCsv[columnName].plot(color='blue', label=f'{columnName} Price')
    operation.plot.scatter(x="Day", y=1 , c = ['r' if x else 'g' for x in (operation.ActivesOperated > 0)], label='Operation',ax=ax)
    plt.title('Extracción de datos del activo')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del activo')
    plt.legend()
    plt.show() 
    plt.clf()

def extractOperations(result):
    info = {
        "Nganancias": 0,
        "Nrecueraciones": 0,
        "Nperdidas": 0,
        "gananciasTotales": 0,
        "perdidasTotales":0,
        "ganancias medias": 0,
        "perdidas medias": 0,
        "ganancias maximas": 0,
        "perdidas maximas": 0,
    }
    for operation in result:
        if operation["Benefit"] < 0: 
            info["Nperdidas"] += 1
            info["perdidasTotales"] += operation["Benefit"]
            if(operation["Benefit"]<info["perdidas maximas"]):info["perdidas maximas"]=operation["Benefit"]

        elif operation["Benefit"] > 0: 
            info["Nganancias"] += 1
            info["gananciasTotales"] += operation["Benefit"]
            if(operation["Benefit"]>info["ganancias maximas"]):info["ganancias maximas"]=operation["Benefit"]

        else: 
            info["Nrecueraciones"] += 1
    info["Nganancias"]  += 1 if result[-1]["Actives"] > 0 else 0
    lastGananciaPotencial = (result[-1]["Actives"] * result[-1]["CurrentPrice"])
    if(lastGananciaPotencial>info["ganancias maximas"]):info["ganancias maximas"]=lastGananciaPotencial
    info["ganancias medias"] = (info["gananciasTotales"] + lastGananciaPotencial)/ info["Nganancias"] 
    info["perdidas medias"] = info["perdidasTotales"] / info["Nperdidas"]
    df = pd.DataFrame(result)
    df['Datetime'] = pd.to_datetime(df['Day'])
    df = df.set_index('Datetime')
    df = df.sort_index(ascending=True)
    print(info)
    return df


def extractResult(dataOrigin, chromosome):
    if(len(chromosome)!=17):
        print("error in individual chromosome!")
        return None

    myInd = Individuo(chromosome,3,4,None,dataOrigin,10000,1000)
    t1 = time.time()
    fit = myInd.computeFitness(dataOrigin,True)
    t2 = time.time()
    print(t2-t1)
    print(fit)
    parsedOperationHistorical = []
    
    lastAction = None
    lastActionDate = None
    for operation in myInd.historicalOperations:
        operation["Last Action Date"] = lastActionDate
        parsedOperationHistorical.append({
            "Day": operation["Day"],
            "Action": operation["Action"],
            "Last Action": lastAction,
            "Last Action Date": lastActionDate,
            "CurrentBudget": operation["CurrentBudget"],
            "Actives":  operation['CurrentStockActives'], 
            "ActivesOperated": operation['ActivesOperated'], 
            "CurrentPrice": operation["CurrentPrice"],
            "PotentialMoney": operation["CurrentBudget"] +(operation['CurrentStockActives'] * operation["CurrentPrice"])
        }
        )
        lastAction = operation["Action"]
        lastActionDate = operation["Day"]


    lastPotentialMoney = 10000
    for index, operation in enumerate(parsedOperationHistorical):
        operation["Benefit"] = operation['PotentialMoney'] - lastPotentialMoney
        lastPotentialMoney = operation['PotentialMoney'] 
    return parsedOperationHistorical

if __name__ == '__main__':
    filenameOrigin = "./dataToTrainEvaluation/AAPL-2022-05-19.pkl"
    chromosome = '11100100011000001'

    dataOrigin = pd.read_pickle(filenameOrigin)
    dataOrigin = dataOrigin.sort_index()
    trainingRatio = 0.7
    dfTrain , dfTest = splitContinous(trainingRatio, dataOrigin)
    dataToUse = dfTest
    result = extractResult(dataToUse, chromosome)
    dfOperations = extractOperations(result)
    plotOperations(dataToUse, "close", dfOperations)
    

    