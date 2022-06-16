import sys
import pandas as pd

def getFitness(orderList, interest = 0.05 , lastActionPrice=None, currentActions = 0 , fastFormat = True, modeSellShort = False): 
    """
    Computes the performance of actions taken in an operation window
    
    Parameters
    ----------
    orderList: list of action performed in an operation window
    interest [Optional]: Indicates the interest rate, default is 0.05 
    lastActionPrice [Optional]: Indicates the last value of the active, default = None (Gets the last value from order list, but not of the whole window, case hold at the end)
    currentActions [Optional]: Indicates the number of hold actives, default is 0
    fastFormat [Optional]: Indicates if we return only the fit value or more information, default is True
    modeSellShort [Optional]: Set to true if operating in that way, default is False
    
    Returns
    -------
    Float: BenefitsFromSells - LossFromBuys + currentValueOfHoldActives 
    or
    {
        "Return": BenefitsFromSells - LossFromBuys + currentValueOfHoldActives,
        "High Benefit": max(Sell),
        "High Lost": min(Buy) ,
        "nOrders": Total Actives operated Sells + Buys,
        "nOperations": Total orders placed in 'packages' ie: Buy 2 actives at once,
        "nBuys": Total Buys orders in packages,
        "nSells":Total Sells orders in packages,
        "Interest": Shows wether there is or not interest rates to be applied,
    }
    """
    if lastActionPrice is None:
        lastActionPrice = orderList[-1]['ActionPrice']
    # - as negative means sell so the income is positive, sell 32 for 10 its represented as -32, 10 and means 320 income
    if fastFormat: return ((lastActionPrice * currentActions) + sum(-(order['Actions'] * order['ActionPrice']) for order in orderList))
    lo, hi = sys.maxsize, -sys.maxsize-1
    nOperations, grossIncome, nBuys, nSells = 0, 0, 0, 0  #grossIncome = sum(order['Actions'] * order['ActionPrice'] for order in orderList)
    for order in orderList:
        value = -order['Actions'] * order['ActionPrice'] 
        grossIncome += value
        nOperations += abs(order['Actions'])
        lo, hi = min(value,lo), max(value,hi)
        if value < 0:
            nBuys += 1
        elif value > 0:
            nSells += 1
            
    nOrders = len(orderList)
    if (nBuys + nSells) != nOrders: print("WARNING! SOME OPERATION HAS 0 VALUE OR SOMETHING ELSE WENT WRONG!!")
    if modeSellShort: 
        return {
            "Return": -((lastActionPrice * currentActions) + grossIncome),
            "High Benefit": -lo,
            "High Lost": -hi,
            "nOrders":nOrders,
            "nOperations":nOperations,
            "nBuys":nSells,
            "nSells":nBuys,
            "Interest":interest,
            "LeftActives":currentActions,
            "CurrentActivePrice":lastActionPrice
        }
    return {
        "Return": (lastActionPrice * currentActions) + grossIncome,
        "High Benefit": hi,
        "High Lost": lo,
        "nOrders":nOrders,
        "nOperations":nOperations,
        "nBuys":nBuys,
        "nSells":nSells,
        "Interest":interest,
        "LeftActives":currentActions,
        "CurrentActivePrice":lastActionPrice
    }
    

def getOrderList(dfOrders, referencePrice='close'):
    """
    Extract a list with the sells/buys and the price of the active at that price
    
    Parameters
    ----------
    df: dataframe with OPEN HIGH LOW CLOSE VOLUME ... ExtractedIndices ... columns
    referencePrice: Selects the price [OPEN HIGH LOW CLOSE] to take into account

    Returns
    -------
    List: [{
        "Actions": +For#Buys | -For#Sells,
        "ActionPrice": ActivePrice
        },...] 
    """
    ordersList = list(filter(lambda x: x['orders'] != 0, dfOrders[["orders",referencePrice]].to_dict("records")))
    for order in ordersList:
        order['Actions'] = order.pop('orders')
        order['ActionPrice'] = order.pop(referencePrice)
    return ordersList

def getCurrentActivePriceFromDF(df, referencePrice='close'):
    """
    Extract the last value from the operation window 

    Parameters
    ----------
    df: dataframe with OPEN HIGH LOW CLOSE VOLUME ... ExtractedIndices ... columns
    referencePrice: Selects the price [OPEN HIGH LOW CLOSE] to take into account

    Returns
    -------
    Float: Last active price
    """
    return df[referencePrice].iat[-1]

if __name__ == '__main__':
    #In order to try, you must download the data and try, example of format:
    filename = "Historical+Orders.pkl"
    currentActions = 667
    dfOrders = pd.read_csv(filename)
    orders = getOrderList(dfOrders)
    print(orders)
    print(getFitness(orders, currentActions=currentActions, lastActionPrice=getCurrentActivePriceFromDF(dfOrders), fastFormat=False))

