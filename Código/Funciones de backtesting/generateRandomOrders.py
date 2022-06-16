import pandas as pd
import numpy as np
import math
import random as rng
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
 
def generate_random_orders(historicalDf):
    historicalDf['orders'] = np.random.randint(-1, 2, historicalDf.shape[0])
    randomOrders = np.random.randint(1, 99, historicalDf.shape[0])
    historicalDf['orders'] = historicalDf['orders'] * randomOrders
    return historicalDf

def generate_random_realistic_orders_by_budget(historicalDf, Budget, referencePrice='close'):
    currentActions = 0
    #indicates if there will be a buy, a sell or nothing
    historicalDf['orders'] = np.random.randint(-1, 2, historicalDf.shape[0])
    for i, row in historicalDf.iterrows():
        #change to budget > 0 if you can buy 0.x actives
        if (row['orders'] > 0 and Budget > row[referencePrice]):
            maxActions = math.floor(Budget/row[referencePrice])
            investment = rng.randint(1, maxActions+1)
            if(Budget - (investment * row[referencePrice]) >= 0):
                currentActions += investment
                historicalDf.at[i,'orders'] =  investment
                Budget -= investment * row[referencePrice]
            else:
                historicalDf.at[i,'orders'] = 0
        elif (row['orders'] < 0 and currentActions > 0):
            sell = rng.randint(1, currentActions+1)
            currentActions -= sell
            Budget += sell * row[referencePrice]
            historicalDf.at[i,'orders'] = -sell
        else:
            historicalDf.at[i,'orders'] = 0
        if Budget < 0: print("You cant borrow money, so something must gone wrong!")
    return [historicalDf, currentActions, Budget]
        
def simulateOrderGeneratorBot(historicalDf, InitialBudget, referencePrice='close'):
    currentActions = 0
    currentBudget = InitialBudget
    robotPosibleActions = {
        -1:"Sell",
        0:"Hold",
        1:"Buy"
    }
    for i, row in historicalDf.iterrows():
        actionPlaced = False
        actionRng = rng.randint(-1,1)
        robotAction = robotPosibleActions[actionRng]
        #case buy
        if robotAction == "Buy":
            #check we can buy, change to > 0 if wou can buy 0.x of an active
            if currentBudget > row[referencePrice]:
                #check how much can we buy
                maxActivesToBuy = math.floor(currentBudget/row[referencePrice])
                #select a valid quantity to buy
                activesToBuy =  rng.randint(1, maxActivesToBuy)
                #update our budget
                currentBudget -= activesToBuy * row[referencePrice]
                #save the action we performed
                historicalDf.at[i,'orders'] =  activesToBuy
                #update the number of actives we have to sell
                currentActions += activesToBuy
                actionPlaced = True
        #case sell
        elif robotAction == "Sell":
            #check we can sell some actives
            if currentActions > 0:
                activesToSell = rng.randint(1, currentActions)
                currentActions -= activesToSell
                currentBudget += activesToSell * row[referencePrice]
                historicalDf.at[i,'orders'] = -activesToSell
                actionPlaced = True
        if not actionPlaced:
            historicalDf.at[i,'orders'] = 0
    return [historicalDf, currentActions, currentBudget]

def simulatesellShortOrderGeneratorBot(historicalDf, referencePrice='close'):
    currentActions = 0
    currentBudget = 0
    maxActivesToBuy = 100
    robotPosibleActions = {
        -1:"BuyShort",
        0:"Hold",
        1:"SellShort"
    }
    for i, row in historicalDf.iterrows():
        actionPlaced = False
        actionRng = rng.randint(-1,1)
        robotAction = robotPosibleActions[actionRng]
        #case buy
        if robotAction == "BuyShort":  
            #select a valid quantity to buy
            activesToBuy =  rng.randint(1, maxActivesToBuy)
            #update our budget
            currentBudget += activesToBuy * row[referencePrice]
            #save the action we performed
            historicalDf.at[i,'orders'] =  activesToBuy
            #update the number of actives we have to sell
            currentActions += activesToBuy
            actionPlaced = True
        #case sell
        elif robotAction == "SellShort":
            #check we can sell some actives
            if currentActions > 0:
                activesToSell = rng.randint(1, currentActions)
                currentActions -= activesToSell
                currentBudget -= activesToSell * row[referencePrice]
                historicalDf.at[i,'orders'] = -activesToSell
                actionPlaced = True
        if not actionPlaced:
            historicalDf.at[i,'orders'] = 0
    return [historicalDf, currentActions, currentBudget]

def buyandholdExtreme(filename, initialBudget):
    df = pd.read_pickle(filename)
    #df = generate_random_orders(df)
    dftrain, dftest = splitContinous(0.7,df)
    activesBought = math.floor(initialBudget/dftest.iloc[0]['close'])
    leftbudget = (initialBudget/dftest.iloc[0]['close']) - math.floor(initialBudget/dftest.iloc[0]['close'])
    moneySell = activesBought*dftest.iloc[-1]['close']
    return moneySell + leftbudget
if __name__ == '__main__':

    #In order to try, you must download the data and try, example of format:
    FILENAMES = [
        "dataToTrainEvaluation/F-2022-05-19.pkl",
        "dataToTrainEvaluation/BBVA-2022-06-14.pkl",
        "dataToTrainEvaluation/MSFT-2022-05-16.pkl",
        "dataToTrainEvaluation/AAPL-2022-05-19.pkl"
    ]

    initialBudget = 10000

    TickerResult = [[buyandholdExtreme(filename, initialBudget)-initialBudget] for filename in FILENAMES]
    print(TickerResult)
    fig, ax = plt.subplots()
    # Creating plot
    plt.boxplot(TickerResult)
    ax.set_xticklabels(["FORD", "BBVA", "MSFT", "AAPL"])
    plt.axhline(y = 0, color = 'r', linestyle = '-')
    plt.title("RESULTADOS ESTRATEGIAS COMPRA ALEATORIA")
    plt.ylabel("GANANCIA (€)")
    plt.xlabel("TICKER")   # show plot
    plt.show()
    # df, currentActions, Budget = simulateOrderGeneratorBot(df, initialBudget)
    # df.to_pickle('Historical+Orders.pkl')
    
    
