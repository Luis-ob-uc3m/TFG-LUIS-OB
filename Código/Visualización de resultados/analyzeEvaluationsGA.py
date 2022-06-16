import json
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from time import sleep
import statistics

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

if __name__ == '__main__':
    Tickers = ["F", "BBVA", "MSFT", "AAPL"]
    Frequency  = "15 MIN"
    Indicador  = "SMA"


    evaluations = []
    for Ticker in Tickers:
        filenameOriginalResults = F"./DATA/{Ticker}/{Frequency}/{Indicador}_OutputResults.json"
        results = json.load(open(filenameOriginalResults))["Results"]
        for result in results:
            evaluations.append(result["nIterations"])

    print(statistics.mean(evaluations))