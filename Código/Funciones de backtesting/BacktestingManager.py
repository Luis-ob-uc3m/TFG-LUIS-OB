from Backtesting import *
from generateRandomOrders import *
import statistics
import matplotlib.pyplot as plt

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

def performExperiment(df, initialBudget, referencePrice='close'):
    dfOrders, currentActions, currentBudget = simulateOrderGeneratorBot(df, initialBudget)
    #dfOrders[['date','open','low','high','close','volume','orders']].to_csv('Actions.csv')
    orders = getOrderList(dfOrders, referencePrice)
    return getFitness(orders, currentActions=currentActions, lastActionPrice=getCurrentActivePriceFromDF(dfOrders), fastFormat=True)
   #print(getFitness(orders, currentActions=currentActions, lastActionPrice=getCurrentActivePriceFromDF(dfOrders), fastFormat=False))


def computeRNGexperiments(filename, nExperiments):
    df = pd.read_pickle(filename)
    dfTrain, dfTest = splitContinous(0.7, df)
    initialBudget = 10000
    referencePrice = 'close'
    #print(Result)
    #print(statistics.mean(Result))    
    return [performExperiment(dfTest, initialBudget, referencePrice) for x in range(nExperiments)]

if __name__ == '__main__':
    #In order to try, you must download the data and try, example of format:
    FILENAMES = [
        "dataToTrainEvaluation/F-2022-05-19.pkl",
        "dataToTrainEvaluation/BBVA-2022-06-14.pkl",
        "dataToTrainEvaluation/MSFT-2022-05-16.pkl",
        "dataToTrainEvaluation/AAPL-2022-05-19.pkl"
    ]
    TickerResult = [computeRNGexperiments(filename, 1000)  for filename in FILENAMES]
    fig, ax = plt.subplots()
    # Creating plot
    plt.boxplot(TickerResult)
    ax.set_xticklabels(["FORD", "BBVA", "MSFT", "AAPL"])
    plt.axhline(y = 0, color = 'r', linestyle = '-')
    plt.title("RESULTADOS ESTRATEGIAS COMPRA ALEATORIA")
    plt.ylabel("GANANCIA (€)")
    plt.xlabel("TICKER")   # show plot
    plt.show()
    