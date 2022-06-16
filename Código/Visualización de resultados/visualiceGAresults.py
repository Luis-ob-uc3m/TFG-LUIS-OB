
import matplotlib.pyplot as plt
import json


#------------------------funciones de plot-----------------------------------
def plotResults(resultados, Parametros):
    """
        Plots all the results

        Parameters 
        ----------
        resultados: results for each experiment
        Parametros: parameters info of each experiment
    """
    resultsToPlot = []
    for resultado in resultados:
        resultsToPlot.append(plotResult(resultado))
    #plt.axhline(y=0, color='r', linestyle='-')  
    plotBestFitness(Parametros, resultsToPlot)
    plt.show()
    #plt.axhline(y=0, color='r', linestyle='-')  
    plotWorstFitness(Parametros, resultsToPlot)
    plt.show()
    plotPS(Parametros, resultsToPlot)
    plt.show()
    plotDS(Parametros, resultsToPlot)
    plt.show()

def plotDS(Parametros, PresionSelectivaResults):
    """
        Plots diversity values

        Parameters 
        ----------
        PresionSelectivaResults: results for each experiment due diversity
        Parametros: parameters info of each experiment
        
    """
    plt.title("Evolucion Diversidad")
    plt.xlabel("Generacion")
    plt.ylabel("Valor de Diversidad")    
    index = 0
    for experimento in  PresionSelectivaResults:
        plt.plot(experimento["Diversidad"],label = f"Experimento {index}" ) #label = Parametros[index]['Descripcion'] )
        index += 1
    plt.legend(loc="upper right")

def plotPS(Parametros, PresionSelectivaResults):
    """
        Plots selective pressure values

        Parameters 
        ----------
        PresionSelectivaResults: results for each experiment due to selective pressure
        Parametros: parameters info of each experiment
        
    """
    plt.title("Evolucion Presion selectiva")
    plt.xlabel("Generacion")
    plt.ylabel("Valor de Presion selectiva")    
    plt.axhline(y=1.5, color='b', linestyle='dotted')  
    plt.axhline(y=1, color='r', linestyle='dotted')  
    index = 0
    for experimento in  PresionSelectivaResults:
        plt.plot(experimento["PresionSelectiva"],label = f"Experimento {index}" ) #label = Parametros[index]['Descripcion'] )
        index += 1
    plt.legend(loc="upper right")

def plotWorstFitness(Parametros, WorstFitnessResults):
    """
        Plots worst fitness values

        Parameters 
        ----------
        PresionSelectivaResults: results for each experiment due worst fit
        Parametros: parameters info of each experiment
        
    """
    plt.title("Evolucion Worst Fitness de Poblacion")
    plt.xlabel("Generacion")
    plt.ylabel("Valor de fitness")
    index = 0
    for experimento in  WorstFitnessResults:
        plt.plot(experimento["valorFitWorst"], label = f"Experimento {index}" ) #label = Parametros[index]['Descripcion'] )
        index += 1
    plt.legend(loc="upper right")

def plotBestFitness(Parametros, BestFitnessResults):
    """
        Plots best fitness values

        Parameters 
        ----------
        PresionSelectivaResults: results for each experiment due best fit
        Parametros: parameters info of each experiment
        
    """
    #plt.axhline(y=0, color='r', linestyle='-')  
    plt.title("Evolucion Best fitness de Poblacion")
    plt.xlabel("Generacion")
    plt.ylabel("Valor de fitness")
    index = 0
    for experimento in BestFitnessResults:
        plt.plot(experimento["valorFitBest"])#, label = f"Experimento {index}" ) #label = Parametros[index]['Descripcion'] )
    plt.legend(loc="upper right")

def plotResult(resultado):
    """
        Extracts all the info to use for plotting

        Parameters 
        ----------
        resultado: All info of result of experiment

        Returns
        ----------
        All info of experiment in a dict
    """
    valorFitBest = []
    valorFitWorst = []
    valorPS = []
    valorDiv = []
    for resultado in resultado["ResultadoObtenido"]:
        valorFitBest.append(float(resultado["BestFitnessOfthisIteration"]))
        valorFitWorst.append(float(resultado["WorstFitnessOfthisIteration"]))
        valorPS.append(float(resultado["PresionSelectiva"]))
        valorDiv.append(float(resultado["Diversidad"]))
    return {
            "valorFitBest": valorFitBest, 
            "valorFitWorst": valorFitWorst,
            "PresionSelectiva": valorPS,
            "Diversidad": valorDiv
            }
    
def loadParameters():
    """
    Opens the params file
    """
    with open("params.json", "r+") as file:
        ParametrosFile = json.load(file)
        return ParametrosFile["Experimentos"]


if __name__ == '__main__':
    Ticker = "AAPL"
    Frequency  = "15 MIN"
    Indicador  = "SMA"
    filename = F"./DATA/{Ticker}/{Frequency}/{Indicador}_OutputResults.json"
    output = json.load(open(filename))
    Parametros = loadParameters()
    #show a plot for documentation and graphical view
    plotResults(output["Results"], Parametros)
    