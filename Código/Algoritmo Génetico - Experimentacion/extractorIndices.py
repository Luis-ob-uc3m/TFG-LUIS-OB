from re import A, T
from matplotlib.font_manager import json_load
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#--------------------------------------------deprecated functions (not used)--------------------------------------------
def extractRSI(df, ventana, append=True):
    newDf = pd.DataFrame()
    newDf['diff'] = df['close'].diff(1)
    newDf['gain'] = newDf['diff'].clip(lower=0)
    newDf['loss'] = newDf['diff'].clip(upper=0).abs()
    newDf['avg_gain'] = newDf['gain'].rolling(window=ventana, min_periods=ventana).mean()[:ventana+1]
    newDf['avg_loss'] = newDf['loss'].rolling(window=ventana, min_periods=ventana).mean()[:ventana+1]
    # Get WMS averages
    # Average Gains
    for i, row in enumerate(newDf['avg_gain'].iloc[ventana+1:]):
        newDf['avg_gain'].iloc[i + ventana + 1] =\
            (newDf['avg_gain'].iloc[i + ventana] *
            (ventana - 1) +
            newDf['gain'].iloc[i + ventana + 1])\
            / ventana
    # Average Losses
    for i, row in enumerate(newDf['avg_loss'].iloc[ventana+1:]):
        newDf['avg_loss'].iloc[i + ventana + 1] =\
            (newDf['avg_loss'].iloc[i + ventana] *
            (ventana - 1) +
            newDf['loss'].iloc[i + ventana + 1])\
            / ventana
    newDf['rs'] = newDf['avg_gain'] / newDf['avg_loss']
    newDf['rsi'] = 100 - (100 / (1.0 + newDf['rs']))
    if(append):
        df[f'RSI {ventana}'] = newDf['rsi']
    return newDf

def extractSMA(df, ventana, append=True):
    newDf = pd.DataFrame()
    newDf[f'SMA_{ventana}'] = df['close'].rolling(window=ventana).mean()
    if(append):
        df[f'SMA_{ventana}'] = newDf
    return newDf

def extractEMA(df, ventana, append=True):
    newDf = pd.DataFrame()
    #adjust=False si interesados en metodo de calculo recursivo.
    newDf[f'EMA {ventana}'] = df['close'].ewm(span=ventana, adjust=False).mean()
    if(append):
        df[f'EMA {ventana}'] = newDf
    return newDf

def extractMACD(df, ventanaSup, ventanaInf, signal, append=True):
    #adjust=False si interesados en metodo de calculo recursivo.
    #k = df['Close'].ewm(span=ventanaInf, adjust=False, min_periods=ventanaInf).mean()
    #d = df['Close'].ewm(span=ventanaSup, adjust=False, min_periods=ventanaSup).mean()
    #macd = k - d
    return df.ta.macd(close='close', fast=ventanaInf, slow=ventanaSup, signal=signal, append=append)

#------------------------------------------------------------------------------------------------------------------------------------

def extractSMALib(df, parametersIndicators):
    """
        Extracts the sma based on ta_pandas lib

        Parameters 
        ----------
        parametersIndicators: parameters used for sma

        Returns 
        ----------
        New dateframe with sma column
    """
    dfNew = df.copy()
    for parameterIndicator in parametersIndicators:
        dfNew.ta.sma(close='close', length=parameterIndicator['ventana'], offset=parameterIndicator['offset'], append=True)
    return dfNew
def extractSWMALib(df, parametersIndicators):
    """
        Extracts the swma based on ta_pandas lib

        Parameters 
        ----------
        parametersIndicators: parameters used for swma

        Returns 
        ----------
        New dateframe with swma column
    """
    dfNew = df.copy()
    for parameterIndicator in parametersIndicators:
        dfNew.ta.swma(close='close', length=parameterIndicator['ventana'], asc=parameterIndicator['asc'], offset=parameterIndicator['offset'], append=True)
    return dfNew
def extractEMALib(df, parametersIndicators):
    """
        Extracts the ema based on ta_pandas lib

        Parameters 
        ----------
        parametersIndicators: parameters used for ema

        Returns 
        ----------
        New dateframe with ema column
    """
    dfNew = df.copy()
    for parameterIndicator in parametersIndicators:
        dfNew.ta.ema(close='close',  length=parameterIndicator['ventana'], offset=parameterIndicator['offset'], append=True)
    return dfNew
def extractRSILib(df, parametersIndicators):
    """
        Extracts the rsi based on ta_pandas lib

        Parameters 
        ----------
        parametersIndicators: parameters used for rsi

        Returns 
        ----------
        New dateframe with rsi column
    """
    dfNew = df.copy()
    for parameterIndicator in parametersIndicators:
        dfNew.ta.rsi(close='close',  length=parameterIndicator['ventana'], drift=parameterIndicator['drift'], offset=parameterIndicator['offset'], append=True)
    return dfNew


def extractAllTAIndicators(df, parametersIndicators):
    """
    Extract most of the ta-lib indicators from OHLCV data
    
    Parameters
    ----------
    df: dataframe with OPEN HIGH LOW CLOSE VOLUME columns
    parametersIndicators: dictionary with the parameters of the indices extracted

    Returns
    -------
    Updated dataframe with the indices extracted
    """
    dfNew = df.copy()
    dfNew.ta.sma(close='close', length=parametersIndicators['SMA']['ventana'], offset=parametersIndicators['SMA']['offset'], append=True)
    dfNew.ta.swma(close='close', length=parametersIndicators['SWMA']['ventana'], asc=parametersIndicators['SWMA']['asc'], offset=parametersIndicators['SWMA']['offset'], append=True)
    dfNew.ta.ema(close='close',  length=parametersIndicators['EMA']['ventana'], offset=parametersIndicators['EMA']['offset'], append=True)
    dfNew.ta.rsi(close='close',  length=parametersIndicators['RSI']['ventana'], drift=parametersIndicators['RSI']['drift'], offset=parametersIndicators['RSI']['offset'], append=True)
    dfNew.ta.stochrsi(close='close', length=parametersIndicators['STOCHRSI']['ventana'], rsi_length=parametersIndicators['STOCHRSI']['ventanaRSI'], k=parametersIndicators['STOCHRSI']['k'], d=parametersIndicators['STOCHRSI']['d'], offset=parametersIndicators['STOCHRSI']['offset'], append=True)
    dfNew.ta.stoch(high='high', low='low', close='close', k=parametersIndicators['STOCH']['k'], d=parametersIndicators['STOCH']['d'], smooth_k=parametersIndicators['STOCH']['ksmooth'], offset=parametersIndicators['STOCH']['offset'] , append=True)
    dfNew.ta.macd(close='close', fast=parametersIndicators['MACD']['ventanaInf'], slow=parametersIndicators['MACD']['ventanaSup'], signal=parametersIndicators['MACD']['signal'], offset=parametersIndicators['MACD']['offset'], append=True)
    dfNew.ta.mom(close='close', length=parametersIndicators['MOM']['ventana'], offset=parametersIndicators['MOM']['offset'], append=True)
    dfNew.ta.eom(high='high', low='low', close='close',volume='volume' , length=parametersIndicators['EOM']['ventana'], divisor=parametersIndicators['EOM']['divisor'], drift=parametersIndicators['EOM']['drift'],  offset=parametersIndicators['EOM']['offset'], append=True)
    dfNew.ta.cci(high='high', low='low', close='close', length=parametersIndicators['CCI']['ventana'], c=parametersIndicators['CCI']['c'], offset=parametersIndicators['CCI']['offset'], append=True)
    dfNew.ta.rma(close='close', length=parametersIndicators['RMA']['ventana'], offset=parametersIndicators['RMA']['offset'], append=True)
    dfNew.ta.roc(close='close', length=parametersIndicators['ROC']['ventana'], scalar=parametersIndicators['ROC']['scalar'],offset=parametersIndicators['ROC']['offset'], append=True)
    dfNew.ta.aberration(high='high', low='low', close='close', length=parametersIndicators['ABERRATION']['ventana'], atr_length=parametersIndicators['ABERRATION']['ventanaATR'], offset=parametersIndicators['ABERRATION']['offset'], append=True)
    dfNew.ta.accbands(high='high', low='low', close='close', length=parametersIndicators['ACCBANDS']['ventana'], c=parametersIndicators['ACCBANDS']['c'],drift=parametersIndicators['ACCBANDS']['drift'], offset=parametersIndicators['ACCBANDS']['offset'], append=True)
    dfNew.ta.ad(high='high', low='low', close='close',volume='volume', open_='open', offset=parametersIndicators['AD']['offset'], append=True)
    dfNew.ta.adosc(high='high', low='low', close='close',volume='volume', open_='open', fast=parametersIndicators['ADOSC']['fast'], slow=parametersIndicators['ADOSC']['slow'], offset=parametersIndicators['ADOSC']['offset'], append=True)
    dfNew.ta.adx(high='high', low='low', close='close', length=parametersIndicators['ADX']['ventana'], lensig=parametersIndicators['ADX']['signal'], scalar=parametersIndicators['ADX']['scalar'],drift=parametersIndicators['ADX']['drift'], offset=parametersIndicators['ADX']['offset'], append=True)
    dfNew.ta.alma(close='close', length=parametersIndicators['ALMA']['ventana'], sigma=parametersIndicators['ALMA']['sigma'], distribution_offset=parametersIndicators['ALMA']['distribution'], offset=parametersIndicators['ALMA']['offset'], append=True)
    dfNew.ta.amat(close='close', fast=parametersIndicators['AMAT']['fast'], slow=parametersIndicators['AMAT']['slow'], lookback=parametersIndicators['AMAT']['lookback'], offset=parametersIndicators['AMAT']['offset'], append=True)
    dfNew.ta.ao(high='high', low='low', fast=parametersIndicators['AO']['fast'], slow=parametersIndicators['AO']['slow'], offset=parametersIndicators['AO']['offset'], append=True)
    dfNew.ta.aobv(close='close', volume='volume', fast=parametersIndicators['AOBV']['fast'], slow=parametersIndicators['AOBV']['slow'], max_lookback=parametersIndicators['AOBV']['maxLookback'], min_lookback=parametersIndicators['AOBV']['minLookback'], offset=parametersIndicators['AOBV']['offset'], append=True)
    dfNew.ta.apo(close='close', fast=parametersIndicators['APO']['fast'], slow=parametersIndicators['APO']['slow'], offset=parametersIndicators['APO']['offset'], append=True)
    dfNew.ta.aroon(high='high', low='low',  length=parametersIndicators['AROON']['ventana'], scalar=parametersIndicators['AROON']['scalar'], offset=parametersIndicators['AROON']['offset'], append=True)
    dfNew.ta.atr(high='high', low='low', close='close', length=parametersIndicators['ATR']['ventana'], drift=parametersIndicators['ATR']['drift'], offset=parametersIndicators['ATR']['offset'], append=True)
    dfNew.ta.bbands(close='close', length=parametersIndicators['BBANDS']['ventana'], std=parametersIndicators['BBANDS']['std'] , ddof=parametersIndicators['BBANDS']['ddof'] , offset=parametersIndicators['BBANDS']['offset'] , append=True)
    dfNew.ta.bias(close='close', length=parametersIndicators['BIAS']['ventana'], offset=parametersIndicators['BIAS']['offset'],append=True)
    dfNew.ta.bop(open='open', high='high', low='low', close='close', scalar=parametersIndicators['BOP']['scalar'], offset=parametersIndicators['BOP']['offset'], append=True)
    dfNew.ta.brar(open='open', high='high', low='low', close='close', length=parametersIndicators['BRAR']['ventana'], scalar=parametersIndicators['BRAR']['scalar'], drift=parametersIndicators['BRAR']['drift'], offset=parametersIndicators['BRAR']['offset'], append=True)
    dfNew.ta.cdl_pattern(open='open', high='high', low='low', close='close', name= 'all', scalar=parametersIndicators['CDL_PATTERN']['scalar'], offset=parametersIndicators['CDL_PATTERN']['offset'], append=True)
    dfNew.ta.cdl_z(open='open', high='high', low='low', close='close',  length=parametersIndicators['CDL_Z']['ventana'], full=parametersIndicators['CDL_Z']['full'], ddof=parametersIndicators['CDL_Z']['ddof'], offset=parametersIndicators['CDL_Z']['offset'], append=True)
    dfNew.ta.cfo(close='close', length=parametersIndicators['CFO']['ventana'], scalar=parametersIndicators['CFO']['scalar'], drift=parametersIndicators['CFO']['drift'], offset=parametersIndicators['CFO']['offset'], append=True)
    dfNew.ta.cg(close='close', length=parametersIndicators['CG']['ventana'], offset=parametersIndicators['CG']['offset'], append=True)
    dfNew.ta.chop(high='high', low='low', close='close', length=parametersIndicators['CHOP']['ventana'], atr_length=parametersIndicators['CHOP']['ventanaATR'], ln=parametersIndicators['CHOP']['ln'], scalar=parametersIndicators['CHOP']['scalar'], drift=parametersIndicators['CHOP']['drift'], offset=parametersIndicators['CHOP']['offset'], append=True)
    dfNew.ta.cmf(high='high', low='low', close='close', volume='volume', open_='open', length=parametersIndicators['CMF']['ventana'], offset=parametersIndicators['CMF']['offset'], append=True)
    dfNew.ta.cmo(close='close', length=parametersIndicators['CMO']['ventana'], scalar=parametersIndicators['CMO']['scalar'], drift=parametersIndicators['CMO']['drift'], offset=parametersIndicators['CMO']['offset'], append=True)
    dfNew.ta.coppock(close='close', length=parametersIndicators['COPPOCK']['ventana'], fast=parametersIndicators['COPPOCK']['fast'], slow=parametersIndicators['COPPOCK']['slow'], offset=parametersIndicators['COPPOCK']['offset'], append=True)
    dfNew.ta.cti(close='close', length=parametersIndicators['CTI']['ventana'], offset=parametersIndicators['CTI']['offset'], append=True) 
    dfNew.ta.decay(close='close', length=parametersIndicators['DECAY']['ventana'], mode='linear', offset=parametersIndicators['DECAY']['offset'], append=True)
    dfNew.ta.decay(close='close', length=parametersIndicators['DECAYEXP']['ventana'], mode='exp', offset=parametersIndicators['DECAYEXP']['offset'], append=True)
    dfNew.ta.decreasing(close='close', length=parametersIndicators['DECREASING']['ventana'], strict=parametersIndicators['DECREASING']['strict'], asint=parametersIndicators['DECREASING']['asint'], percent=parametersIndicators['DECREASING']['percent'], drift=parametersIndicators['DECREASING']['drift'], offset=parametersIndicators['DECREASING']['offset'], append=True)
    dfNew.ta.dema(close='close', length=parametersIndicators['DEMA']['ventana'], offset=parametersIndicators['DEMA']['offset'],append=True)
    dfNew.ta.dm(high='high', low='low', length=parametersIndicators['DM']['ventana'], drift=parametersIndicators['DM']['drift'], offset=parametersIndicators['DM']['offset'], append=True)
    dfNew.ta.donchian(high='high', low='low', lower_length=parametersIndicators['DONCHIAN']['ventanaInf'], upper_length=parametersIndicators['DONCHIAN']['ventanaSup'], offset=parametersIndicators['DONCHIAN']['offset'],append=True)
    dfNew.ta.dpo(close='close', length=parametersIndicators['DPO']['ventana'], centered=parametersIndicators['DPO']['centered'], offset=parametersIndicators['DPO']['offset'], append=True)
    dfNew.ta.ebsw(close='close', length=parametersIndicators['EBSW']['ventana'], bars=parametersIndicators['EBSW']['bars'], offset=parametersIndicators['EBSW']['offset'], append=True)
    dfNew.ta.efi(close='close',volume='volume', length=parametersIndicators['EFI']['ventana'], drift=parametersIndicators['EFI']['drift'], offset=parametersIndicators['EFI']['offset'], append=True)
    dfNew.ta.entropy(close='close', length=parametersIndicators['ENTROPY']['ventana'], base=parametersIndicators['ENTROPY']['base'], offset=parametersIndicators['ENTROPY']['offset'], append=True)
    dfNew.ta.er(close='close', length=parametersIndicators['ER']['ventana'], drift=parametersIndicators['ER']['drift'], offset=parametersIndicators['ER']['offset'], append=True)
    dfNew.ta.eri(high='high', low='low', close='close', length=parametersIndicators['ERI']['ventana'], offset=parametersIndicators['ERI']['offset'], append=True)
    dfNew.ta.fisher(high='high', low='low', length=parametersIndicators['FISHER']['ventana'], signal=parametersIndicators['FISHER']['signal'], offset=parametersIndicators['FISHER']['offset'], append=True)
    dfNew.ta.fwma(close='close', length=parametersIndicators['FWMA']['ventana'], asc=parametersIndicators['FWMA']['asc'], offset=parametersIndicators['FWMA']['offset'], append=True)
    dfNew.ta.ha(open='open', high='high', low='low', close='close',  offset=parametersIndicators['HA']['offset'], append=True)
    dfNew.ta.hilo(high='high', low='low', close='close', high_length=parametersIndicators['HILO']['ventanaHigh'], low_length=parametersIndicators['HILO']['ventanaLow'], offset=parametersIndicators['HILO']['offset'], append=True)
    dfNew.ta.hl2(high='high', low='low', offset=parametersIndicators['HL2']['offset'], append=True)
    dfNew.ta.hlc3(high='high', low='low', close='close', offset=parametersIndicators['HLC3']['offset'], append=True)
    dfNew.ta.hma(close='close', length=parametersIndicators['HMA']['ventana'], offset=parametersIndicators['HMA']['offset'],append=True)
    dfNew.ta.hwc(close='close', na=parametersIndicators['HWC']['na'], nb=parametersIndicators['HWC']['nb'], nc=parametersIndicators['HWC']['nc'], nd=parametersIndicators['HWC']['nd'], scalar=parametersIndicators['HWC']['scalar'], channel_eval=parametersIndicators['HWC']['channel'], offset=parametersIndicators['HWC']['offset'],append=True)
    dfNew.ta.hwma(close='close', na=parametersIndicators['HWMA']['na'], nb=parametersIndicators['HWMA']['nb'], nc=parametersIndicators['HWMA']['nc'],  offset=parametersIndicators['HWMA']['offset'],append=True)
    dfNew.ta.ichimoku(high='high', low='low', close='close',  tenkan=parametersIndicators['ICHIMOKU']['tenkan'], kijun=parametersIndicators['ICHIMOKU']['kijun'], senkou=parametersIndicators['ICHIMOKU']['senkou'], include_chikou=parametersIndicators['ICHIMOKU']['chikou'], offset=parametersIndicators['ICHIMOKU']['offset'], append=True)
    dfNew.ta.increasing(close='close', length=parametersIndicators['INCREASING']['ventana'], strict=parametersIndicators['INCREASING']['strict'], asint=parametersIndicators['INCREASING']['asint'], percent=parametersIndicators['INCREASING']['percent'], drift=parametersIndicators['INCREASING']['drift'], offset=parametersIndicators['INCREASING']['offset'],append=True)
    dfNew.ta.inertia(close='close', high='high', low='low', length=parametersIndicators['INERTIA']['ventana'], rvi_length=parametersIndicators['INERTIA']['ventanaRVI'], scalar=parametersIndicators['INERTIA']['scalar'], refined=parametersIndicators['INERTIA']['refined'], thirds=parametersIndicators['INERTIA']['thirds'], drift=parametersIndicators['INERTIA']['drift'], offset=parametersIndicators['INERTIA']['offset'],append=True)
    dfNew.ta.jma(close='close', length=parametersIndicators['JMA']['ventana'], phase=parametersIndicators['JMA']['phase'], offset=parametersIndicators['JMA']['offset'], append=True)
    dfNew.ta.kama(close='close', length=parametersIndicators['KAMA']['ventana'], fast=parametersIndicators['KAMA']['fast'], slow=parametersIndicators['KAMA']['slow'], drift=parametersIndicators['KAMA']['drift'], offset=parametersIndicators['KAMA']['offset'], append=True)
    dfNew.ta.kc(high='high', low='low', close='close', length=parametersIndicators['KC']['ventana'], scalar=parametersIndicators['KC']['scalar'],  offset=parametersIndicators['KC']['offset'], append=True)
    dfNew.ta.kdj(high='high', low='low', close='close', length=parametersIndicators['KDJ']['ventana'], signal=parametersIndicators['KDJ']['signal'], offset=parametersIndicators['KDJ']['offset'], append=True)
    dfNew.ta.kst(close='close', roc1=parametersIndicators['KST']['roc1'], roc2=parametersIndicators['KST']['roc2'], roc3=parametersIndicators['KST']['roc3'], roc4=parametersIndicators['KST']['roc4'], sma1=parametersIndicators['KST']['sma1'], sma2=parametersIndicators['KST']['sma2'], sma3=parametersIndicators['KST']['sma3'], sma4=parametersIndicators['KST']['sma4'], signal=parametersIndicators['KST']['signal'], drift=parametersIndicators['KST']['drift'], offset=parametersIndicators['KST']['offset'], append=True)
    dfNew.ta.kurtosis(close='close', length=parametersIndicators['KURTOSIS']['ventana'], offset=parametersIndicators['KURTOSIS']['offset'], append=True)
    dfNew.ta.kvo(high='high', low='low', close='close', volume='volume', fast=parametersIndicators['KVO']['fast'], slow=parametersIndicators['KVO']['slow'], signal=parametersIndicators['KVO']['signal'], drift=parametersIndicators['KVO']['drift'], offset=parametersIndicators['KVO']['offset'], append=True)
    dfNew.ta.linreg(close='close', length=parametersIndicators['LINREG']['ventana'], offset=parametersIndicators['LINREG']['offset'], append=True)
    dfNew.ta.log_return(close='close', length=parametersIndicators['LOG_RETURN']['ventana'], cumulative=parametersIndicators['LOG_RETURN']['cumulative'], offset=parametersIndicators['LOG_RETURN']['offset'],append=True)
    dfNew.ta.long_run(fast='fast', slow='slow', length=parametersIndicators['LONG_RUN']['ventana'], offset=parametersIndicators['LONG_RUN']['offset'], append=True)
    dfNew.ta.mad(close='close', length=parametersIndicators['MAD']['ventana'], offset=parametersIndicators['MAD']['offset'],append=True)
    dfNew.ta.massi(high='high', low='low', fast=parametersIndicators['MASSI']['fast'], slow=parametersIndicators['MASSI']['slow'], offset=parametersIndicators['MASSI']['offset'], append=True)
    dfNew.ta.mcgd(close='close', length=parametersIndicators['MCGD']['ventana'], offset=parametersIndicators['MCGD']['offset'], c=parametersIndicators['MCGD']['c'], append=True)
    dfNew.ta.median(close='close', length=parametersIndicators['MEDIAN']['ventana'], offset=parametersIndicators['MEDIAN']['offset'], append=True)
    dfNew.ta.mfi(high='high', low='low', close='close', volume='volume', length=parametersIndicators['MFI']['ventana'], drift=parametersIndicators['MFI']['drift'], offset=parametersIndicators['MFI']['offset'], append=True)
    dfNew.ta.midpoint(close='close', length=parametersIndicators['MIDPOINT']['ventana'], offset=parametersIndicators['MIDPOINT']['offset'], append=True)
    dfNew.ta.midprice(high='high', low='low', length=parametersIndicators['MIDPRICE']['ventana'],  offset=parametersIndicators['MIDPRICE']['offset'], append=True)
    dfNew.ta.natr(high='high', low='low', close='close', length=parametersIndicators['NATR']['ventana'], scalar=parametersIndicators['NATR']['scalar'],drift=parametersIndicators['NATR']['drift'], offset=parametersIndicators['NATR']['offset'], append=True)
    dfNew.ta.nvi(close='close', volume='volume', length=parametersIndicators['NVI']['ventana'], initial=parametersIndicators['NVI']['initial'], offset=parametersIndicators['NVI']['offset'], append=True)
    dfNew.ta.obv(close='close', volume='volume', offset=parametersIndicators['OBV']['offset'],append=True)
    dfNew.ta.ohlc4(open='open', high='high', low='low', close='close', offset=parametersIndicators['OHLC4']['offset'], append=True)
    dfNew.ta.pdist(open='open', high='high', low='low', close='close',drift=parametersIndicators['PDIST']['drift'], offset=parametersIndicators['PDIST']['offset'], append=True)
    dfNew.ta.percent_return(close='close', length=parametersIndicators['PERCENT_RETURN']['ventana'], cumulative=parametersIndicators['PERCENT_RETURN']['cumulative'], offset=parametersIndicators['PERCENT_RETURN']['offset'], append=True)
    dfNew.ta.pgo(high='high', low='low', close='close', length=parametersIndicators['PGO']['ventana'], offset=parametersIndicators['PGO']['offset'],append=True)
    dfNew.ta.ppo(close='close', fast=parametersIndicators['PPO']['fast'], slow=parametersIndicators['PPO']['slow'], signal=parametersIndicators['PPO']['signal'], scalar=parametersIndicators['PPO']['scalar'],  offset=parametersIndicators['PPO']['offset'] , append=True)
    dfNew.ta.psar(high='high', low='low', close='close', af0=parametersIndicators['PSAR']['af0'], af=parametersIndicators['PSAR']['af'], max_af=parametersIndicators['PSAR']['afMax'], offset=parametersIndicators['PSAR']['offset'], append=True)
    dfNew.ta.psl(close='close', open='open', length=parametersIndicators['PSL']['ventana'], scalar=parametersIndicators['PSL']['scalar'],drift=parametersIndicators['PSL']['drift'], offset=parametersIndicators['PSL']['offset'], append=True)
    dfNew.ta.pvi(close='close', volume='volume', length=parametersIndicators['PVI']['ventana'], initial=parametersIndicators['PVI']['initial'], offset=parametersIndicators['PVI']['offset'], append=True)
    dfNew.ta.pvo(volume='volume', fast=parametersIndicators['PVO']['fast'], slow=parametersIndicators['PVO']['slow'], signal=parametersIndicators['PVO']['signal'], scalar=parametersIndicators['PVO']['scalar'], offset=parametersIndicators['PVO']['offset'], append=True)
    dfNew.ta.pvol(close='close', volume='volume', offset=parametersIndicators['PVOL']['offset'],append=True)
    dfNew.ta.pvr(close='close', volume='volume')
    dfNew.ta.pvt(close='close', volume='volume', drift=parametersIndicators['PVT']['drift'], offset=parametersIndicators['PVT']['offset'], append=True)
    dfNew.ta.pwma(close='close', length=parametersIndicators['PWMA']['ventana'], asc=parametersIndicators['PWMA']['asc'], offset=parametersIndicators['PWMA']['offset'], append=True)
    dfNew.ta.qqe(close='close', length=parametersIndicators['QQE']['ventana'], smooth=parametersIndicators['QQE']['smooth'], factor=parametersIndicators['QQE']['factor'], drift=parametersIndicators['QQE']['drift'], offset=parametersIndicators['QQE']['offset'], append=True)
    dfNew.ta.qstick(open='open', close='close', length=parametersIndicators['QSTICK']['ventana'], offset=parametersIndicators['QSTICK']['offset'], append=True)
    dfNew.ta.quantile(close='close', length=parametersIndicators['QUANTILE']['ventana'], q=parametersIndicators['QUANTILE']['q'], offset=parametersIndicators['QUANTILE']['offset'], append=True)
    dfNew.ta.rsx(close='close', length=parametersIndicators['RSX']['ventana'], drift=parametersIndicators['RSX']['drift'], offset=parametersIndicators['RSX']['offset'], append=True)
    dfNew.ta.rvgi(open='open', high='high', low='low', close='close', length=parametersIndicators['RVGI']['ventana'], swma_length=parametersIndicators['RVGI']['ventanaSWMA'], offset=parametersIndicators['RVGI']['offset'], append=True)
    dfNew.ta.rvi(close='close', high='high', low='low', length=parametersIndicators['RVI']['ventana'], scalar=parametersIndicators['RVI']['scalar'], refined=parametersIndicators['RVI']['refined'], thirds=parametersIndicators['RVI']['thirds'], drift=parametersIndicators['RVI']['drift'], offset=parametersIndicators['RVI']['offset'], append=True)
    dfNew.ta.short_run(fast='fast', slow='slow', length=parametersIndicators['SHORT_RUN']['ventana'], offset=parametersIndicators['SHORT_RUN']['offset'], append=True)
    dfNew.ta.sinwma(close='close', length=parametersIndicators['SINWMA']['ventana'], offset=parametersIndicators['SINWMA']['offset'], append=True)
    dfNew.ta.skew(close='close', length=parametersIndicators['SKEW']['ventana'], offset=parametersIndicators['SKEW']['offset'], append=True)
    dfNew.ta.slope(close='close', length=parametersIndicators['SLOPE']['ventana'], as_angle=parametersIndicators['SLOPE']['angle'], to_degrees=parametersIndicators['SLOPE']['degree'], vertical=parametersIndicators['SLOPE']['vertical'], offset=parametersIndicators['SLOPE']['offset'], append=True)
    dfNew.ta.smi(close='close', fast=parametersIndicators['SMI']['fast'], slow=parametersIndicators['SMI']['slow'], signal=parametersIndicators['SMI']['signal'], scalar=parametersIndicators['SMI']['scalar'], offset=parametersIndicators['SMI']['offset'], append=True)
    dfNew.ta.squeeze(high='high', low='low', close='close', bb_length=parametersIndicators['SQUEEZE']['ventanaBB'], bb_std=parametersIndicators['SQUEEZE']['desviacionBB'], kc_length=parametersIndicators['SQUEEZE']['ventanaKC'], kc_scalar=parametersIndicators['SQUEEZE']['scalarKC'], mom_length=parametersIndicators['SQUEEZE']['ventanaMOM'], mom_smooth=parametersIndicators['SQUEEZE']['smoothMOM'], use_tr=parametersIndicators['SQUEEZE']['tr'],  offset=parametersIndicators['SQUEEZE']['offset'], append=True)
    dfNew.ta.squeeze_pro(high='high', low='low', close='close', bb_length=parametersIndicators['SQUEEZEPRO']['ventanaBB'], bb_std=parametersIndicators['SQUEEZEPRO']['desviacionBB'], kc_length=parametersIndicators['SQUEEZEPRO']['ventanaKC'], kc_scalar=parametersIndicators['SQUEEZEPRO']['scalarKC'], kc_scalar_wide=parametersIndicators['SQUEEZEPRO']['scalarWideKC'], kc_scalar_normal=parametersIndicators['SQUEEZEPRO']['scalarNormalKC'], kc_scalar_narrow=parametersIndicators['SQUEEZEPRO']['scalarNarrowKC'], mom_length=parametersIndicators['SQUEEZEPRO']['ventanaMOM'], mom_smooth=parametersIndicators['SQUEEZEPRO']['smoothMOM'], use_tr=parametersIndicators['SQUEEZEPRO']['tr'],  offset=parametersIndicators['SQUEEZEPRO']['offset'], append=True)
    dfNew.ta.ssf(close='close', length=parametersIndicators['SSF']['ventana'], poles=parametersIndicators['SSF']['poles'], offset=parametersIndicators['SSF']['offset'], append=True)
    dfNew.ta.stc(close='close', tclength=parametersIndicators['STC']['ventanaTCL'], fast=parametersIndicators['STC']['fast'], slow=parametersIndicators['STC']['slow'], factor=parametersIndicators['STC']['factor'], offset=parametersIndicators['STC']['offset'], append=True)
    dfNew.ta.stdev(close='close', length=parametersIndicators['STDEV']['ventana'], ddof=parametersIndicators['STDEV']['ddof'],  offset=parametersIndicators['STDEV']['offset'], append=True)
    dfNew.ta.supertrend(high='high', low='low', close='close', length=parametersIndicators['SUPERTREND']['ventana'],multiplier=parametersIndicators['SUPERTREND']['multiplier'], offset=parametersIndicators['SUPERTREND']['offset'], append=True)
    dfNew.ta.t3(close='close', length=parametersIndicators['T3']['ventana'], a=parametersIndicators['T3']['a'], offset=parametersIndicators['T3']['offset'], append=True)
    dfNew.ta.td_seq(close='close', asint=parametersIndicators['TD_SEQ']['asint'], offset=parametersIndicators['TD_SEQ']['offset'],append=True)
    dfNew.ta.tema(close='close', length=parametersIndicators['TEMA']['ventana'], offset=parametersIndicators['TEMA']['offset'], append=True)
    dfNew.ta.thermo(high='high', low='low', length=parametersIndicators['THERMO']['ventana'], long=parametersIndicators['THERMO']['long'],short=parametersIndicators['THERMO']['short'], drift=parametersIndicators['THERMO']['drift'] , offset=parametersIndicators['THERMO']['offset'], append=True)
    dfNew.ta.tos_stdevall(close='close', length=parametersIndicators['TOS_STDEVALL']['ventana'], stds=parametersIndicators['TOS_STDEVALL']['stds'], ddof=parametersIndicators['TOS_STDEVALL']['ddof'], offset=parametersIndicators['TOS_STDEVALL']['offset'], append=True)
    dfNew.ta.trima(close='close', length=parametersIndicators['TRIMA']['ventana'], offset=parametersIndicators['TRIMA']['offset'],append=True)
    dfNew.ta.trix(close='close', length=parametersIndicators['TRIX']['ventana'], signal=parametersIndicators['TRIX']['signal'], scalar=parametersIndicators['TRIX']['scalar'], drift=parametersIndicators['TRIX']['drift'], offset=parametersIndicators['TRIX']['offset'], append=True)
    dfNew.ta.true_range(high='high', low='low', close='close', drift=parametersIndicators['TRUE_RANGE']['drift'], offset=parametersIndicators['TRUE_RANGE']['offset'], append=True)
    dfNew.ta.tsi(close='close', fast=parametersIndicators['TSI']['fast'], slow=parametersIndicators['TSI']['slow'], signal=parametersIndicators['TSI']['signal'], scalar=parametersIndicators['TSI']['scalar'],drift=parametersIndicators['TSI']['drift'], offset=parametersIndicators['TSI']['offset'],append=True)
    dfNew.ta.ttm_trend(high='high', low='low', close='close', length=parametersIndicators['TTM_TREND']['ventana'], offset=parametersIndicators['TTM_TREND']['offset'], append=True)
    dfNew.ta.ui(close='close', length=parametersIndicators['UI']['ventana'], scalar=parametersIndicators['UI']['scalar'], offset=parametersIndicators['UI']['offset'], append=True)
    dfNew.ta.uo(high='high', low='low', close='close', fast=parametersIndicators['UO']['fast'], medium=parametersIndicators['UO']['medium'], slow=parametersIndicators['UO']['slow'], fast_w=parametersIndicators['UO']['fastW'], medium_w=parametersIndicators['UO']['mediumW'], slow_w=parametersIndicators['UO']['slowW'], drift=parametersIndicators['UO']['drift'], offset=parametersIndicators['UO']['offset'], append=True)
    dfNew.ta.variance(close='close', length=parametersIndicators['VARIANCE']['ventana'], ddof=parametersIndicators['VARIANCE']['ddof'], offset=parametersIndicators['VARIANCE']['offset'], append=True)
    dfNew.ta.vhf(close='close', length=parametersIndicators['VHF']['ventana'], drift=parametersIndicators['VHF']['drift'], offset=parametersIndicators['VHF']['offset'], append=True)
    dfNew.ta.vidya(close='close', length=parametersIndicators['VIDYA']['ventana'], drift=parametersIndicators['VIDYA']['drift'], offset=parametersIndicators['VIDYA']['offset'],append=True)
    dfNew.ta.vortex(high='high', low='low', close='close', length=parametersIndicators['VORTEX']['ventana'], drift=parametersIndicators['VORTEX']['drift'], offset=parametersIndicators['VORTEX']['offset'], append=True)
    dfNew.ta.vp(close='close', volume='volume', width=parametersIndicators['VP']['ventana'], append=True)
    dfNew.ta.vwap(high='high', low='low', close='close', volume='volume', anchor=parametersIndicators['VWAP']['anchor'], offset=parametersIndicators['VWAP']['anchor'], append=True)
    dfNew.ta.vwma(close='close', volume='volume', length=parametersIndicators['VWMA']['ventana'], offset=parametersIndicators['VWMA']['offset'],append=True)
    dfNew.ta.wcp(high='high', low='low', close='close',  offset=parametersIndicators['WCP']['offset'], append=True)
    dfNew.ta.willr(high='high', low='low', close='close', length=parametersIndicators['WILLR']['ventana'],  offset=parametersIndicators['WILLR']['offset'], append=True)
    dfNew.ta.wma(close='close', length=parametersIndicators['WMA']['ventana'], asc=parametersIndicators['WMA']['asc'], offset=parametersIndicators['WMA']['offset'], append=True)
    dfNew.ta.zlma(close='close', length=parametersIndicators['ZLMA']['ventana'],  offset=parametersIndicators['ZLMA']['offset'], append=True)
    dfNew.ta.zscore(close='close', length=parametersIndicators['ZSCORE']['ventana'], std=parametersIndicators['ZSCORE']['std'], offset=parametersIndicators['ZSCORE']['offset'], append=True)
    return dfNew



if __name__ == '__main__':
    filename = "./DatosDescargados/MSFT-2022-03-21.pkl"
    parametersFile = './parameters.json'
    parametersIndicators = json_load(parametersFile)
    df = pd.read_pickle(filename)  
    # ventana = 14
    # fast=26
    # slow=12
    # signal=9
    # extractSMA(df, ventana)
    # extractEMA(df, ventana)
    # extractRSI(df,ventana
    # extractMACD(df,fast,slow,signal)

    df = extractAllTAIndicators(df,parametersIndicators)
    df.to_pickle("HistoricalData.pkl")
    
    # df = pd.read_csv(filename)
    # df = df[['date', 'close']]
    # date_time_str = '2021-01-01 00:00:00'
    # df = df[df['date'] > date_time_str]
    
    # df.plot(x ='date', y='close', kind = 'line', label='MSFT')
    # plt.gca().invert_xaxis()
    # plt.show()
    # plt.clf()
    # df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    # print (df.dtypes)
    # exp1 = df.close.ewm(span=12, adjust=False).mean()
    # exp2 = df.close.ewm(span=26, adjust=False).mean()
    # macd = exp1-exp2
    # exp3 = macd.ewm(span=9, adjust=False).mean()
    # plt.plot(df.date, macd, label='AMD MACD', color = '#EBD2BE')
    # plt.plot(df.date, exp3, label='Signal Line', color='#E5A4CB')
    # plt.legend(loc='upper left')
    # plt.show()