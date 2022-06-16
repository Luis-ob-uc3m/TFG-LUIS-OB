import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from time import sleep


def loadDataAsCSV(filename = "MSFT-2022-01-24.csv"):
    csvFile = pd.read_csv("./DatosDescargados/"+filename, index_col=0, parse_dates=True)
    csvFile.index = pd.to_datetime(csvFile.index)
    return csvFile

def plotData(dataCsv, columnName):
    dataCsv[columnName].plot(color='blue', label='Precio de cierre')
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
    
if __name__ == '__main__':
    #In order to try, you must download the data and try, example of format:
    priceDataLoad = loadDataAsCSV("MSFT-2022-06-06.csv")
    print(priceDataLoad.info())
    plotData(priceDataLoad, "close")
    plotNext(priceDataLoad, "open", "close")
    plotCandleAsLine(priceDataLoad)
    candlestickChart(priceDataLoad)
    
