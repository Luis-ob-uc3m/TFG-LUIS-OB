
from statistics import mean
import matplotlib.pyplot as plt
import json


    
def loadParameters():
    """
    Opens the params file
    """
    with open("params.json", "r+") as file:
        ParametrosFile = json.load(file)
        return ParametrosFile["Experimentos"]

def extractFitsfOLDER(results, TICKERS, INDICATORS):
    FOLDERS = ["5 MIN", "15 MIN" , "30 MIN"]
    FITS = {
            "5 MIN":[],
            "15 MIN":[],
            "30 MIN":[]
        }
    for ticker in TICKERS:
        for folder in FOLDERS:
            for indicator in INDICATORS:
                for resultado in results[ticker][folder][indicator]:
                    FITS[folder].append(resultado[0])
    return FITS

def computeMeasures(results, TICKERS, FOLDERS, INDICATORS):
    tp = 0 
    fp = 0 
    tn = 0 
    fn = 0 
    for ticker in TICKERS:
        for folder in FOLDERS:
            for indicator in INDICATORS:
                for resultado in RESULTS[ticker][folder][indicator]:
                    if (resultado[0] > 0 and resultado[2] > 0 ):
                        tp += 1
                    elif (resultado[0] > 0 and resultado[2] < 0 ):
                        fp += 1
                    elif (resultado[0] < 0 and resultado[2] > 0 ):
                        fn += 1
                    elif (resultado[0] < 0 and resultado[2] < 0 ):
                        tn += 1


    return tp, fp, tn, fn 

if __name__ == '__main__':
    #Ticker = "AAPL"
    #Frequency  = "15 MIN"
    #Indicador  = "SMA"
    #filename = F"./DATA/{Ticker}/{Frequency}/{Indicador}_OutputResults.json"
    #output = json.load(open(filename))
    
    Parametros = loadParameters()
    TICKERS = ["F", "AAPL", "MSFT", "BBVA"]
    FOLDERS = ["5 MIN", "15 MIN" , "30 MIN"]
    INDICATORS = ["SMA" ,"EMA"]
    RESULTS = {
        "F": {
            "5 MIN": {
                "SMA":[],
                "EMA":[]
            }
            ,
            "15 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "30 MIN": {
                "SMA":[],
                "EMA":[]
            }
        },
        "AAPL": {
            "5 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "15 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "30 MIN": {
                "SMA":[],
                "EMA":[]
            }
        },
        "MSFT": {
            "5 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "15 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "30 MIN": {
                "SMA":[],
                "EMA":[]
            }
        },
        "BBVA": {
            "5 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "15 MIN": {
                "SMA":[],
                "EMA":[]
            },
            "30 MIN": {
                "SMA":[],
                "EMA":[]
            }
        },
    }
    for ticker in TICKERS:
        for folder in FOLDERS:
            for indicator in INDICATORS:
                filename = f"./DATA/{ticker}/{folder}/{indicator}_OutputResults.json"
                try:
                    with open(filename, "r+") as file:
                        ResultsJSON = json.load(file)
                        for experimentResult in ResultsJSON["Results"]:
                            RESULTS[ticker][folder][indicator].append([experimentResult["BestFitness"], experimentResult["BestIndividuo"], experimentResult["testResults"]])
                except: pass
    # with open('results.json', 'w') as fp:
    #     json.dump(RESULTS, fp) 
    
    
    for folder in FOLDERS:
        tp, fp, tn, fn = computeMeasures(RESULTS, TICKERS, [folder], ["EMA"])
        print(folder)
        print(tp, fp, tn, fn)

        tp, fp, tn, fn = computeMeasures(RESULTS, TICKERS, [folder], ["SMA"])
        print(folder)
        print(tp, fp, tn, fn)
