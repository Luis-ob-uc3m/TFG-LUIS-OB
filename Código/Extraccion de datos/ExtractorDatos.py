
#apikey = "YOURAPIKEYHERE"
apikey = "YOURAPIKEYHERE"    

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
from datetime import datetime 
from datetime import date
import json
from time import time
import pandas as pd
from itertools import islice

def checkLastDate(lastDateHistorical, lastDate, n=3):
    """
    Checks if the element ``lastDate`` appears ``n`` times in ``lastDateHistorical``

    Parameters
    ----------
    lastDateHistorical: Identificator of the asset whose data is extracted
    lastDate: 
    n: number of times the element must appear (3 by default)

    Returns
    -------
    True if the elemenets appear n times
    """
    if ((len(lastDateHistorical) == 0) or (lastDate is None)): 
        return False
    boolList = (True for dateRecorded in lastDateHistorical if dateRecorded==lastDate)
    return next(islice(boolList, n-1, None), False)

def saveData(assetID, result):
    """
    Takes the extraction result of an asset and adds it to the current data extracted
    if no data is extracted it creates a new file
    The file created has an id such that {assetID}-{Year-Month-Day}.json

    Parameters
    ----------
    assetID: Identificator of the asset whose data is extracted
    result: Json data of the asset
    
    Returns
    -------
    The  json filename that has been updated
    """
    filename = f"./DatosDescargados/{assetID}-{date.today().strftime('%Y-%m-%d')}.json"
    lastDate = None
    try:
        lastDate = datetime.fromisoformat(getLastDateFromJson(filename))
        print("Ultima fecha recogida anteriormente: " + lastDate.strftime('%Y-%m-%d'))
        #add data to json
        with open(filename,"r+") as file:
            current_file_data = json.load(file)
            for validResult in [resultado for resultado in result if lastDate > datetime.fromisoformat(resultado['date'])]:
                current_file_data.append(validResult)
            file.seek(0)
            json.dump(current_file_data, file, indent = 4)
    except:
        overwriteAsJson(result, assetID)
    return filename, lastDate

def saveDataOverwrite(assetID, timeframe, result):
    """
    Takes a file and overwrites its content 
    it saves the corresponding json and csv files
    The file created has an id such that {assetID}-{Year-Month-Day}.json

    Parameters
    ----------
    assetID: Identificator of the asset whose data is extracted
    result: Json data of the asset
    timeframe: Frequency of data extraction ["1min", "5min", "15min", "30min", "1hour"] or historical" to identify the csv format
    
    Returns
    -------
    The  json filename that has been updated
    """
    filenameJSON = overwriteAsJson(result, assetID)
    getDataAsCSV(result, assetID, timeframe)
    return filenameJSON #for csv .csv instead of .json

def overwriteAsJson(result, assetID):
    """
    overwrites the file and saves the data extracted of the asset
    The file created has an id such that {assetID}-{Year-Month-Day}.json

    Parameters
    ----------
    assetID: Identificator of the asset whose data is extracted
    result: Json data of the asset

    Returns
    -------
    The  json filename that has been updated
    """
    
    filename = f"./DatosDescargados/{assetID}-{date.today().strftime('%Y-%m-%d')}.json"
    with open(filename,"w") as file:
        json.dump(result, file)
    return filename

def getDataAsCSV(jsonData, assetID, timeframe="historical"):
    """
    Saves a json into a csv
    The file created has an id such that {assetID}-{Year-Month-Day}.csv

    Parameters
    ----------
    assetID: Identificator of the asset whose data is extracted
    jsonData: Json data of the asset, the data extracted
    timeframe: Frequency of data extraction ["1min", "5min", "15min", "30min", "1hour"] or historical" to identify the csv format
    
    Returns
    -------
    csv filename
    """
    
    filename = f"./DatosDescargados/{assetID}-{date.today().strftime('%Y-%m-%d')}.csv"
    writeHeaders(jsonData[0].keys(),filename)
    saveAsCsv(jsonData, filename, timeframe)
    return filename

def writeHeaders(keys,filename):
    """
    From a json file as an array of dicts, takes the keys of the first element 
    to get the headers of csv

    Parameters
    ----------
    keys: keys of the first element
    filename: Json filename

    Returns
    -------
    void
    """
    keys = list(keys)
    keysInCSV = ",".join(list(keys))
    with open(filename,"w") as file:
        file.write(keysInCSV+'\n')

def saveAsCsv(results, filename, timeframe):
    """
    Saves a json into a csv (from an array of elements) writes the content, not the headers

    Parameters
    ----------
    results: Json result of the data extracted (array of dicts)
    filename: CSV filename
    timeframe: Frequency of data extraction ["1min", "5min", "15min", "30min", "1hour"] or historical" to identify the csv format
    

    Returns
    -------
    filename csv
    """
    with open(filename,"a") as file:
        for result in results:
            values = list(result.values())
            if timeframe == "historical":
                values[11] = values[11].replace(",","-")
            valuesInCSV = ",".join(f"{value}" for value in values)
            file.write(valuesInCSV+'\n')
    return filename

def saveDataAsPickle(assetID, filenameCSV, sort=True):
    """
    Saves a csv as a pickle and sorts it to recent date at the bottom

    Parameters
    ----------
    assetID: Identificator of the asset whose data is extracted
    filenameCSV: CSV filename    
    sort: True by default, Shows that the dataframe would be stored with the recent date at the bottom
    
    Returns
    -------
    filename pickle
    """
    filename = f"./DatosDescargados/{assetID}-{date.today().strftime('%Y-%m-%d')}.pkl"
    csvFile = pd.read_csv(filenameCSV, index_col=0, parse_dates=True)
    csvFile.index = pd.to_datetime(csvFile.index)
    if sort:    
        csvFile = csvFile[::-1]
    csvFile.to_pickle(filename)  
    return filename

def get_jsonparsed_data(url):
    """
    Receive the content of ``url``, parse it as JSON and return the object.

    Parameters
    ----------
    url : str

    Returns
    -------
    dict
    """
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def getHistoricalURL(apikey, assetID, startDate=None, endDate=None):
    """
    Receive the content of ``assetID``, to create an url to extract data.
    startDate and Enddate could be set to establish limits 

    Parameters
    ----------
    apikey: key of FMP API
    assetID: FMP Identificator to extract the data
    startDate: Optional, Upper bound for the data (Must establish a lower bound)
    endDate: Optional, Lower bound for the data (Must establish a upper bound)

    Returns
    -------
    url to request data
    """
    if(not(startDate and endDate)):
        return f"https://financialmodelingprep.com/api/v3/historical-price-full/{assetID}?apikey={apikey}"
    return f"https://financialmodelingprep.com/api/v3/historical-price-full/{assetID}?from={startDate}&to={endDate}&apikey={apikey}"

def getminURL(apikey, assetID, timeframe="1hour", startDate=None, endDate=None):
    """
    Receive the content of ``assetID``, to create an url to extract data based on a data frequency ``timeframe``.
    startDate and Enddate could be set to establish limits 

    Parameters
    ----------
    apikey: key of FMP API
    timeframe: Frequency of the data extracted ["1min", "5min", "15min", "30min", "1hour"]
    assetID: FMP Identificator to extract the data
    startDate: Optional, Upper bound for the data (Must establish a lower bound)
    endDate: Optional, Lower bound for the data (Must establish a upper bound)

    Returns
    -------
    url to request data
    """
    validTimeframes = ["1min", "5min", "15min", "30min", "1hour"]
    #si no es valido retornamos el de 1 hora por defecto
    if(timeframe not in validTimeframes):
        return f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{assetID}?apikey={apikey}"
    if(not(startDate and endDate)):
        return f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{assetID}?apikey={apikey}"
    return f"https://financialmodelingprep.com/api/v3/historical-chart/{timeframe}/{assetID}?from={startDate}&to={endDate}&apikey={apikey}"

def getCurrentPrice(apikey, assetID):
    """
    Receive the content of ``assetID``, to create an url to extract current price

    Parameters
    ----------
    apikey: key of FMP API
    assetID: FMP Identificator to extract the data

    Returns
    -------
    url to request current price
    """
    return f"https://financialmodelingprep.com/api/v3/quote-short/{assetID}?apikey={apikey}"

def getLastDateFromJson(filenameJson):
    """
    Receive a json file and gets the date of the last data extracted (Data is ordered temporally)

    Parameters
    ----------
    fileameJson: name of the file to extract last date

    Returns
    -------
    last date as a string (Year-Month-Day Hour:min:second)
    """
    with open(filenameJson) as json_file:
        data = json.load(json_file)
        return data[-1]['date']

if __name__ == "__main__":
    #-------------------------------------INPUTS-------------------------------------------------------
    assetID = 'BBVA' #input("Introduce the assetID ('MSFT','BUSD')\n")
    #fecha más antigua
    startDate = "2021-01-01" #input("Introduce start date with format YYYY-MM-DD (2013-11-09)")
    #fecha más reciente
    endDate = "2022-04-30" ##input("Introduce end date with format YYYY-MM-DD (2021-06475-04)")
    timeframe = "30min" #input("Introduce the frequency of the data extraction ["1min", "5min", "15min", "30min", "1hour"] or historical")
    #--------------------------------------------------------------------------------------------
    print(f"Extrayendo: {assetID} Desde {startDate}, Hasta {endDate}")
    apicalls = 0
    #the historical data has another format so has to be processed in a different way
    if(timeframe == "historical"):
        urlHistorical = getHistoricalURL(apikey, assetID, startDate, endDate)
        result = get_jsonparsed_data(urlHistorical)['historical']
        print(f"Se han extraido un total de {len(result)} datos")
        saveDataOverwrite(assetID, timeframe, result)
    else:
        filenameJson = None
        lastDate = None
        lastDatesHistorical = []
        print(f"Extrayendo datos con una frecuencia de {timeframe}")
        #extraer el máximo de datos, la api retorna desde el valor más reciente al más profundo
        #en cada iteración establecemos la fecha más reciente como la más vieja de la anterior iteracion
        #ahorramos el número máximo de llamadas a la API (pago), implementamos un controlador para evitar bucles debido a fallos de la API
        while(datetime.fromisoformat(endDate) > datetime.fromisoformat(startDate) and not checkLastDate(lastDatesHistorical,lastDate)):  
            urlHistorical = getminURL(apikey, assetID, timeframe, startDate, endDate)
            print(F"{urlHistorical}")
            result = get_jsonparsed_data(urlHistorical)
            apicalls+=1
            filenameJson, lastDate = saveData(assetID, result)
            #remove hours
            endDate = getLastDateFromJson(filenameJson).split(" ")[0]
            lastDatesHistorical.append(lastDate)
        #uNA VEZ TENEMOS TODOS LOS DATOS, PROCESAMOS CSV
        with open(filenameJson) as json_file:
            data = json.load(json_file)
            print(f"Se han extraido un total de {len(data)} datos")
            filenameCSV = getDataAsCSV(data, assetID, timeframe)
            
            saveDataAsPickle(assetID, filenameCSV)
        print(f"Se han realizado un total de {apicalls} llamadas a la API de FMP")