import random as rng
import numpy as np
import math
from extractorIndices import *
from datetime import datetime


def grayCodeToDecimal(numberInGray):
    """
    Right Shift the number by 1 taking xor with original number

    Parameters 
    ----------
    numberInGray: Number in binary representing a graycoded number (Ex: 1001)

    Returns 
    ----------
    Decimal value (Ex:14)
    """
    n = int(numberInGray, 2)
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n

class Individuo:
    """
    CHROMOSOME:
    ----------------------------------------------------------------------
    | EMARANGE1 | EMARANGE2 | EMARANGE3 | COMPARISIONBITS |
    ----------------------------------------------------------------------
    FUTURE CHROMOSOME:
    ----------------------------------------------------------------------
    | INDICATORSELECTOR | INDICATORRANGE1 | INDICATORSELECTOR | INDICATORRANGE2 |INDICATORSELECTOR | INDICATORRANGE3 | COMPARISIONBITS |
    ----------------------------------------------------------------------
    INDICATORRANGE (EMA|SMA..): GRAY CODED
    COMPARISIONBITS: 1 IF TRUE 0 OTHERWISE
    INDICATORSELECTOR: 0 EMA , 1 SMA ...

    V0: EMARANGE 1 2 4 6 8 9 10

    """
    def __init__(self, chromosome = None, numberOfIndicators = 3, sizeBitsIndicator = 4, fitness = None, data = None, initialBudget=0, pricePerOperation=None,  referencePrice='close', autoComputeFit = False):
        """
        Initiates the object, if no chromosome is passed it generates a random individual (And the subsequent needed classes)
        
        Parameters (Optional)
        ----------
        chromosome: Array Containing the DNA (Codification), If provided should be an string "1000....101"
        numberOfIndicators: Numbers of Technical Indicators used in the codification (Default = 3), 
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification (Default = 4), 
        fitness: Value representing how capable the individual is (Default = None), 
        initialBudget: Budget of each individual to operate with,
        pricePerOperation: Amount of money to exped each time we buy/sell
        data: Temporal data used for choosing actions (Default = None), 
        referencePrice: Label to consider currentPrice (Default = 'close')
        autocomputeFit: Indicates if the fit is computed once the individual is created (Default = false)
        """
        self.fitness = fitness
        if ((chromosome is None) or (len(chromosome) ==0)): 
            self.chromosome = []
            self.generateRandomIndividual(numberOfIndicators, sizeBitsIndicator)
        else: 
            self.chromosome = chromosome
            self.chromosomeLen = len(chromosome)
            self.generateIndividualFromChromosome(numberOfIndicators, sizeBitsIndicator)
        self.chromosomeAsArray = self.getChromosomeAsArray()
        
        self.data = data
        self.initialBudget = initialBudget
        self.currentBudget = initialBudget
        self.pricePerOperation = pricePerOperation
        self.stocksBuy = 0
        
        self.status = None  #Register if the last action is buy/sell (Hold not taken into account)
        self.lastOperationValue = 0

        self.operationsPerformed = 0
        self.numberofbuys = 0
        self.numberofsells = 0
        self.numberOfPositiveOperations = 0
        self.gananciaMaxima = initialBudget
        self.perdidaMaxima = initialBudget
        self.referencePrice = referencePrice
        self.computedIndicatorsForCurrentDay = []
        self.historicalComparision = []
        self.historicalOperations = []
        if data is not None: 
            self.currentPrice = data[referencePrice]
            self.computeIndicators()
        else: self.currentPrice = None
        if autoComputeFit: self.computeFitness()

    def generateIndividualFromChromosome(self, numberOfIndicators, sizeBitsIndicator):
        """
        Generates the individual objects (Indicators and comparision) from current chromosome saved as string
    
        Parameters 
        ----------
        numberOfIndicators: Numbers of Technical Indicators used in the codification
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification 

        Returns 
        ----------
        Returns new individual chromosome (Array ob objects)
        """
        position = 0
        createIndicator = []
        createCombination = []
        listIndicators = []
        comparision = None
        lastRange = 1
        stepBase = 3
        for index, dna in enumerate(self.chromosome):
            if index < numberOfIndicators*sizeBitsIndicator:
                createIndicator.append(int(dna))
                if ((index + 1) % sizeBitsIndicator == 0):
                    listIndicators.append(Indicator(position, lastRange, sizeBitsIndicator, stepBase, createIndicator))
                    createIndicator = []
                    lastRange += (2**sizeBitsIndicator)*(stepBase**position)
                    position += 1
            else: createCombination.append(int(dna))
        comparision = Comparision(numberOfIndicators, createCombination)
        listIndicators.append(comparision)
        self.chromosome = listIndicators
        return self.chromosome
        

    def generateRandomIndividual(self, numberOfIndicators, sizeBitsIndicator):
        """
        Generates a new random binary (gray-coded for indicators) chromosome based on the scheme
    
        Parameters 
        ----------
        numberOfIndicators: Numbers of Technical Indicators used in the codification (Default = 3), 
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification (Default = 4), 
        
        Returns 
        ----------
        Updates individual chromosomeLen and chromosome
        Returns the chromosome
        """
        
        ncomparions = 2*numberOfIndicators -1
        self.chromosomeLen = numberOfIndicators * sizeBitsIndicator + ncomparions
        #Generate random 1 and 0 for the indicators
        lastRange = 1
        #Cromosome = Indicator1-...-IndicatorN-Comarision
        #CreateTheIndicator
        stepBase = 3
        for indicatorIndex in range(numberOfIndicators):
            indicator = Indicator(indicatorIndex, lastRange, sizeBitsIndicator, stepBase)
            self.chromosome.append(indicator)
            lastRange += (2**sizeBitsIndicator)*(stepBase**indicatorIndex )
        #Create The Comparision
        self.chromosome.append(Comparision(numberOfIndicators))
        return self.chromosome
    
    def computeIndicators(self, ProvidedData=None):
        """
        Computes indicators values for a dataframe, based on indicators stored in chromosome.
    
        Parameters 
        ----------
        data: OHLC-V data to compute indicators (If not provided is taken from individual current data)
        
        Returns 
        ----------
        Updates the field and returns the new dataframe
        """
        sma = []
        ema = []
        if ProvidedData is None: ProvidedData = self.data
        for indicator in self.chromosome:
            if isinstance(indicator, Indicator):    
                sma.append({"ventana":indicator.decimalValue,"offset":0})
        self.data = extractSMALib(ProvidedData, sma)
        return self.data

    def computeComparision(self, currentPrice = None):
        """
        Checks the number of comparisions that are correct between the comparision gen and indicators values
    
        Parameters 
        ----------
        currentPrice: Active price to compute some comparisions (If not provided is taken from individual current price)
        
        Returns 
        ----------
        Updates individual currentPrice
        Returns boolean indicating if everything has been checked properly (All conditions must be meet)
        """
        if currentPrice is None: currentPrice = self.currentPrice
        if self.computedIndicatorsForCurrentDay is None: 
            print("You must compute indicators values") 
            return None
        if isinstance(self.chromosome[-1], Comparision):   
            return self.chromosome[-1].computeComparision(self.computedIndicatorsForCurrentDay, currentPrice)
        for gen in self.chromosome:
            if  isinstance(gen, Comparision):   
                return gen.computeComparision(self.computedIndicatorsForCurrentDay, currentPrice) 
        print("No comparision found!") 
        return None
        
    def checkManualIndicatorComputation(self, dataCurrentDayIndex):
        """
        Extracts the sma manually
    
        Parameters 
        ----------
        dataCurrentDay: DataframeRow with all the information computed
        
        Returns 
        ----------
        Updates individual currentPrice and saves the IndicatorsCurrentValue
        Returns an array with the values of the indicators selected at chromosome
        """
        indicators = [indicator.coputeValueAuxiliar(dataCurrentDayIndex, self.data) for indicator in self.chromosome if isinstance(indicator, Indicator)]
        if sum([math.isnan(x) for x in indicators]):
            self.computedIndicatorsForCurrentDay=indicators
        return self.computedIndicatorsForCurrentDay 

    def computeIndicatorsCurrentDay(self, dataCurrentDay):
        """
        Extracts the values from the indicators chosen at the chromosome
    
        Parameters 
        ----------
        dataCurrentDay: DataframeRow with all the information computed
        
        Returns 
        ----------
        Updates individual currentPrice and saves the IndicatorsCurrentValue
        Returns an array with the values of the indicators selected at chromosome
        """
        self.computedIndicatorsForCurrentDay = [indicator.coputeValue(dataCurrentDay) for indicator in self.chromosome if isinstance(indicator, Indicator)]
        self.currentPrice = dataCurrentDay[self.referencePrice]
        return self.computedIndicatorsForCurrentDay

    def getIndicatorsNames(self):
        """
        Returns 
        ----------
        Array with IndicatorGeneralName_IndicatorSelectedName (Ex: SMA_16)
        """
        return [indicator.name + '_' + str(indicator.decimalValue) for indicator in self.chromosome if isinstance(indicator, Indicator)]

    def chooseAction(self, nActivesOperated):
        """
        Decides which action do based on current condition and previous Actions
        If condition true and not buy --> buy,
        If condition false and bought --> sell

        Parameters 
        ----------
        nActivesOperated: Numbers of actives to buy/sell

        Returns 
        ----------
        Action to perform (Hold, buy or sell)
        """
        comparision = self.computeComparision()
        self.historicalComparision.append(comparision)
        if (comparision and self.status != "buy"):
            if(self.currentBudget < nActivesOperated * self.currentPrice): return "hold"
            self.status = "buy"
            return "buy"
        elif ((not comparision) and self.status != "sell" and self.stocksBuy>0):
            self.status = "sell"
            return "sell"
        return "hold"
    
    def resetFit(self, df=None):
        """
        Resets the individual so it can be evaluated
        """
        if df is not None: self.data = df
        self.fitness = 0
        self.currentBudget = self.initialBudget
        self.stocksBuy = 0
        self.status = None  #Register if the last action is buy/sell (Hold not taken into account)
        self.operationsPerformed = 0
        self.numberofbuys=0
        self.numberofsells = 0
        self.computedIndicatorsForCurrentDay = []
        self.historicalComparision = []
        self.lastOperationValue = 0        
        self.numberOfPositiveOperations = 0
        self.gananciaMaxima = self.initialBudget
        self.perdidaMaxima = self.initialBudget
        self.computeIndicators()

    def computeFitness(self, data=None, GethistoricalOperations = True):
        """
        Computes the performance of the individual by operating in a time period.

        Parameters 
        ----------
        data: Dataset where to operate (Optional, if not provided uses saved one)
        GethistoricalOperations: saves information about each operation performed

        Returns 
        ----------
        Fitness value as net benefit: (currentBudget + (stocksBuy * currentPrice)) - initialBudget
        """
        if data is None: 
            data = self.data
            if self.data is None: 
                print("Individual Has no data to act!") 
                return None
        #For each time make Actions
        maximumSMA = 200
        for indexRow, rowInfo in enumerate(self.data.iterrows()): 
            if(indexRow < maximumSMA): 
                continue # AVOID NAN VALUES
            dateIndex = rowInfo[0]
            row = rowInfo[1] 
            self.computeIndicatorsCurrentDay(row)
            if sum([math.isnan(x) for x in self.computedIndicatorsForCurrentDay]):
                self.checkManualIndicatorComputation(dateIndex)
            #get how many stocks are we going to operate with
            nActivesOperated = math.floor(self.pricePerOperation/self.currentPrice)
            previousAction = self.status 
            action = self.chooseAction(nActivesOperated)
            if action == "buy": 
                nActivesOperated = nActivesOperated * -1
                self.stocksBuy += abs(nActivesOperated)
            elif action == "sell":
                #you cannot sell more than you have
                nActivesOperated = min(abs(nActivesOperated), self.stocksBuy)
                if nActivesOperated == 0: 
                    #if prices increase hughely, as not adaptative invest, check the inverse 
                    nActivesOperated =  math.floor(self.currentPrice/self.pricePerOperation)
                    nActivesOperated = min(abs(nActivesOperated), self.stocksBuy)
                self.stocksBuy -= abs(nActivesOperated)
            else: nActivesOperated = 0 

            #if make operation save info
            if nActivesOperated != 0 and  action != "hold": 
                if action == "sell":  
                    self.numberofsells += 1
                    if ((nActivesOperated * self.currentPrice) - self.lastOperationValue>0):
                        self.numberOfPositiveOperations += 1
                elif action == "buy": 
                    self.numberofbuys += 1
                    if ((nActivesOperated * self.currentPrice) - self.lastOperationValue < 0):
                        self.numberOfPositiveOperations += 1
                self.lastOperationValue = nActivesOperated * self.currentPrice
                self.operationsPerformed += 1
               

            if self.currentBudget > self.gananciaMaxima :
                self.gananciaMaxima = self.currentBudget
            
            if self.currentBudget < self.perdidaMaxima :
                self.perdidaMaxima = self.currentBudget


            self.currentBudget += nActivesOperated * self.currentPrice
            if GethistoricalOperations and action != "hold": 
                self.historicalOperations.append(
                    {
                        "Action":action,
                        "Day":row.name.strftime("%Y-%m-%d %H:%M:%S"), #row['date'],
                        "ActivesOperated":nActivesOperated,
                        "CurrentPrice": self.currentPrice,
                        'IndicadorCortoPlazo': self.computedIndicatorsForCurrentDay[0],
                        'IndicadorMedioPlazo': self.computedIndicatorsForCurrentDay[1],
                        'IndicadorLargoPlazo': self.computedIndicatorsForCurrentDay[2],
                        "CurrentBudget": self.currentBudget,
                        "CurrentStockActives": self.stocksBuy,
                        "PotentialBudget":self.currentBudget + (self.stocksBuy * self.currentPrice)
                    }
                )
        self.fitness = (self.currentBudget + (self.stocksBuy * self.currentPrice)) - self.initialBudget
        return self.fitness
        
    def getInfo(self):
        """
        Returns 
        ----------
        Individual info as dict
        """
        return {
            "ChromosomeADN": str(self),
            "Fitness" : self.fitness,
            "ChromosomeGens" : self.chromosome,
            "ChromosomeLen" : self.chromosomeLen,
            "HasData": self.data is not None and  self.data.size > 0,
            "CurrentPrice": self.referencePrice,
            "CompuedIndicators":self.computedIndicatorsForCurrentDay,
            "historicalComparision": self.historicalComparision
        }
    def getChromosomeAsArray(self):
        """
        Returns 
        ----------
        Chromosome as array
        """
        return [int(adn) for adn in str(self)]
    def getMeaning(self):
        """
        Returns 
        ----------
        Codification explained by words
        """
        indicators = self.getIndicatorsNames()
        lenComparision = self.chromosome[-1].bitsNeededForComparision
        info = ""
        infoAsDict = {
            0:f"{indicators[0]},{indicators[1]}",
            1:f"{indicators[1]},{indicators[2]}",
            2:f"{indicators[0]},CurrentPrice",
            3:f"{indicators[1]},CurrentPrice",
            4:f"{indicators[2]},CurrentPrice"
        }
        for index,comparisionbit in enumerate(self.chromosomeAsArray[-1-lenComparision:-1]):
            subComp = ">" if comparisionbit else "<="
            info += infoAsDict[index].replace(',',subComp) + "\n"
        return info

    def __str__(self):
        """
        Returns 
        ----------
        Chromosome as string
        """
        return ''.join(map(str, self.chromosome))

class Indicator:
    def __init__(self,indicatorIndex = 0, startingRangeIndex = 1, sizeBitsIndicator = 4, stepBase=3, grayCode = None, ComputeValue = None):
        """
        Initiates the object by creating a randomInstance of it if no grayCode is passed
        Parameters (Optional)
        ----------
        indicatorIndex: Indicates the position at the chromosome (Default = 0)
        startingRangeIndex: Indicators are sorted from lowest to upper ranges, this index avoids collisions and is the maximum value of prefious index (Default = 1)
        sizeBitsIndicator: Size (Bits) of each Technical Indicators used in the codification (Default = 4), 
        stepBase=Size of how indicators grow up its possible values (Steps of stepBase ^ indicatorIndex),
        grayCode = Chromosome gen of the indicator (Default = None, computed randomly), later is translated to decimalValue
        ComputeValue = Actual value of the indicator with current prices (Default = None, computed)
        """
        self.indicatorIndex = indicatorIndex
        if grayCode is  None: self.grayCode = np.random.randint(2, size=sizeBitsIndicator)
        else: self.grayCode = grayCode 
        self.decimalValue = ((stepBase**indicatorIndex) *grayCodeToDecimal(str(self))) + startingRangeIndex
        self.computedValue = ComputeValue
        self.name = 'SMA'
    
    def coputeValue(self, dataCurrentDay):
        """
        Extracts the values from the indicator chosen at the chromosome
    
        Parameters 
        ----------
        dataCurrentDay: DataframeRow with all the information computed
        
        Returns 
        ----------
        Updates and returns the computed value
        """
        self.computedValue = dataCurrentDay[self.name+'_'+str(self.decimalValue)]
        return self.computedValue 

    def coputeValueAuxiliar(self, dataCurrentDayIndex, data):
        Processeddata =  extractSMA(data, self.decimalValue)
        if math.isnan(Processeddata.loc[dataCurrentDayIndex]):
            self.computedValue = Processeddata.loc[dataCurrentDayIndex][self.name+'_'+str(self.decimalValue)]
        return self.computedValue 


    def __str__(self):
        return ''.join(map(str, self.grayCode))

class SMAIndicator(Indicator):
     def __init__(self):
        self.name = 'SMA'

class EMAIndicator(Indicator):
     def __init__(self):
        self.name = 'EMA'

class Comparision:
    """ 
    Imagine we have the following computed values and current price is 20
    if priceIndicator <= --> 0 
    if > -->1
    Ex: Compares 20 with 10, 10 with 30, and 20, 10,30 with currentPrice=20
    ----------------------------------------------------------------------
    | 20 | 10 | 30 | COMPARISIONBITS =  |10001|
    ----------------------------------------------------------------------
    """
    def __init__(self, numberOfIndicators, combiation = None):
        """
        Initiates the object by creating a randomInstance of it if no combination is passed
        Parameters (Optional)
        ----------
        numberOfIndicators: Indicates the numbers of indicators in the chromosome
        combiation: Optional gen value, if not provided is created randomly
        """
        self.bitsNeededForComparision =  2*numberOfIndicators -1
        if combiation is  None: self.combiation = np.random.randint(2, size=self.bitsNeededForComparision)
        else: self.combiation = combiation 
        self.meetComparision = False
        self.numberOFmeet = 0

    def computeComparision(self, Indicators, currentPrice):
        """
        Verifies how the chromosome fits the comparision criteria.
        Value 1 means > , Value 0 means <=
        It compares the indicators in adyacent Positions and current prices as stated in the class description.
    
        Parameters 
        ----------
        Indicators: Array of computed actual values of the selected indicators, in order by indicator position.
        currentPrice: Active current price
        
        Returns 
        ----------
        Updates and number of comparisions meet and if the whole condition is meet 
        Returns true if the whole condition is meet.
        """
        compIndex = 0
        conditionsMeet = 0
        #Compare indicators
        for indicatorIndex in range(len(Indicators)):
            if indicatorIndex == 0: continue
            #check if previous indicators value was 
            check = False
            if ((Indicators[indicatorIndex-1] is not None) and  (Indicators[indicatorIndex] is not None)): check = Indicators[indicatorIndex-1] > Indicators[indicatorIndex]
            if((check and self.combiation[compIndex] == 1 )or (not check and self.combiation[compIndex] == 0)): conditionsMeet += 1
            compIndex += 1
        #Compare with current price
        indicatorIndex = 0
        for combinationIndex in range(compIndex, len(self.combiation)):
            check = False
            if (Indicators[indicatorIndex] is not None): check = Indicators[indicatorIndex] > currentPrice
            if((check and self.combiation[combinationIndex] == 1 )or (not check and self.combiation[combinationIndex] == 0)): conditionsMeet += 1
            indicatorIndex += 1
        self.numberOFmeet = conditionsMeet
        self.meetComparision = conditionsMeet == len(self.combiation)
        return self.meetComparision

    def __str__(self):
        return ''.join(map(str, self.combiation))

if __name__ == '__main__':
    #filename = "./dataTry.pkl"
    #df = pd.read_pickle(filename)
    #myindividuo = Individuo(data = df, initialBudget = 10000, pricePerOperation=1000)
    # myindividuo.getIndicatorsNames()
    # df = myindividuo.computeIndicators()
    # print(df.iloc[200])
    # myindividuo.computeIndicatorsCurrentDay(df.iloc[200])
    # myindividuo.computeComparision()
    # for gen in myindividuo.chromosome:
    #     #print(gen)
    #     # if(isinstance(gen, Indicator)):
    #     #     print(gen.decimalValue)
    #     if(isinstance(gen, Comparision)):
    #         print(gen)
    #         print(gen.numberOFmeet)
    #myindividuo.computeFitness()
    chromosome = '00010000011000111'
    myInd = Individuo(chromosome, numberOfIndicators = 3, sizeBitsIndicator = 4, fitness = None, data = None, initialBudget=0, pricePerOperation=None,  referencePrice='close', autoComputeFit = False)
    print(myInd.getMeaning())