import json 
import csv
import pandas as pd


class TickerParser:
    def __init__(self, jsonDict):
        self.tickers = jsonDict
        self.tickerInList = self.listTickers(self.tickers)
        self.df = self.tickersToDf(self.tickerInList)
        self.csv = self.df.to_csv("./DatosDescargados/results.csv")
        self.saveDictIntoCsv(self.tickers, "./DatosDescargados/results.csv")

    def listTickers(self, tickers):
        date, open, high, low , close, volume = [], [], [], [], [], []
        for ticker in tickers:
            date.append(ticker["date"])
            open.append(ticker["open"])
            high.append(ticker["high"])
            low.append(ticker["low"])
            close.append(ticker["close"])
            volume.append(ticker["volume"])
        return {
            "Date": date ,
            "Open": open,
            "Low": low,
            "High": high,
            "Close": close,
            "Volume": volume
        }

    def tickersToDf(self, tickers):
        df = pd.DataFrame()
        for key in tickers.keys():
            df[key] = tickers[key]
        return df
    
    #without creating a df so we can skip the middle step
    def saveDictIntoCsv(self, dict, fileName):
        headers = dict[0].keys()
        with open(fileName, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for data in dict:
                writer.writerow(data)

    #region derecated as too slow
    def createTickersDf(self, tickers):
        headers = tickers[0].keys()
        df = self.initEmptyDf(headers)
        for ticker in tickers:
            df = self.addTickerInfoToDf(df, ticker)
        return df

    def initEmptyDf(self, headers):
        df = pd.DataFrame()
        for key in headers:
            df[key]=None
        return df

    def addTickerInfoToDf(self, df, ticker):
        return df.append(ticker, ignore_index=True)
    #endregion


if __name__ == '__main__':
    data = None
    #In order to try, you must download the data and try, example of format:
    dataFileToTransform = "MSFT-2022-01-24.json" #input("Whats the file's name to transform?")
    with open("./DatosDescargados/"+dataFileToTransform) as json_file:
        data = json.load(json_file)
    myTickerParser = TickerParser(data)
  