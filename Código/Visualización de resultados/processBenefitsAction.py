import json
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

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
    ax = dataCsv[columnName].plot(color='blue', label=f'{columnName} Price')
    operation.plot.scatter(x="Day",y=0, c = ['r' if x else 'g' for x in (operation.ActivesOperated > 0)], label='Operation',ax=ax)
    plt.title('Extracción de datos del activo')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del activo')
    plt.legend()
    plt.show() 
    plt.clf()





if __name__ == '__main__':
     #In order to try, you must download the data and try, example of format:
    filenameResults = "./AnalyzedResults/dateResultsF.json"
    for result in json.load(open(filenameResults))["Results"]:
        parsedOperationHistorical = []
        for operation in result["OperationsHistorical"]:
            parsedOperationHistorical.append({
                "Day": operation["Day"],
                "CurrentBudget": operation["CurrentBudget"] + (operation["ActivesOperated"] * operation["CurrentPrice"]),
                "Actives":  operation['CurrentStockActives'], 
                "PotentialMoney": operation["CurrentBudget"] + (operation["ActivesOperated"] * operation["CurrentPrice"]) + (operation['CurrentStockActives'] * operation["CurrentPrice"])
            }
            )
        nOperations = len(result["OperationsHistorical"])
        lastPotentialMoney = 10000
        for index, operation in enumerate(parsedOperationHistorical):
            operation["Benefit"] = operation['PotentialMoney'] - lastPotentialMoney
            lastPotentialMoney = operation['PotentialMoney'] 
        
        nbenefit = 0
        nloss = 0
        nrecover = 0
        benefit = 0
        loss = 0
        recover = 0
        lastPotential = 0
        maxBenefit = max([operation["Benefit"] for operation in parsedOperationHistorical])
        minLoss = min([operation["Benefit"] for operation in parsedOperationHistorical])
        
        for operation in parsedOperationHistorical:
            if(operation["Benefit"]>0):
                nbenefit +=1
                benefit += operation["Benefit"]
            elif(operation["Benefit"]<0):
                nloss += 1
                loss += operation["Benefit"]
            else:
                nrecover +=1
            lastPotential = operation["Actives"]
        # print(benefit,  loss)
        # print((benefit + loss )/(nOperations))

        # print(maxBenefit,  minLoss)
        # print(nOperations, nbenefit,  nloss, nrecover)
        # print("------------------")

        # for operation in result["OperationsHistorical"]:
        #     print(operation)
