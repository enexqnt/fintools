from ast import Pass
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import estimation
import scipy

st.title('Lazy portfolio backtester')

n_stocks=st.slider('Select the number of assets: ',min_value=2,max_value=12)

##############################
# DICTIONARY FOR INPUT
##############################
tickers={}
for i in range(0,n_stocks):
     tickers[i] = 0

weights={}
for i in range(0,n_stocks):
     weights[i] = int(1)


##############################
# PLOT FUNCTION
##############################
def plot_g(data,title=None):
    fig = plt.figure()
    plt.plot(data)
    plt.title(title)
    st.pyplot(fig)


##############################
# TICKERS OF ASSETS
##############################
for k, v in tickers.items():
    tickers[k] = st.text_input('Ticker of asset number: ' +str(k), help='Insert the ticker in yahoo finance format',placeholder='WLD.MI')
    st.write(tickers[k])

##############################
# YAHOO FINANCE DATA DOWNLOAD
##############################
tickers=list(tickers.values())
tickers.sort()

data=yf.download(tickers,interval='1d')['Adj Close']
data.dropna(inplace=True)


##############################
# RETURNS COMPUTATION & PLOTTING
##############################
rets=data.pct_change().dropna()
cumrets=(rets+1).cumprod()



plot_g(cumrets,'Cumulative returns')


##############################
## WEIGHTS 
##############################
st.subheader('Define weights of the portfolio')

c=0

for k, v in weights.items():
    weights[k] = st.text_input(tickers[k]+' weight:',placeholder=0.5,key=str(k),value=0)
    st.write(weights[k])
    c+=1

weights=list(weights.values())
weights=[float(i) for i in weights]

st.metric('Sum of weights',sum(weights))

if sum(weights)!=1:
    st.write('You must define weights with sum equal to one!')

else:
    ##############################
    ## HIST PORTFOLIO ANALYSIS 
    ##############################
    st.header('Historical analysis')

    # Rendimento cumulato
    hist=((rets@weights)+1).cumprod()
    plot_g(hist,'Historical cumulative returns')

    ##############################
    ## DRAWDOWN & other metrics
    ##############################
    dd=(hist-hist.cummax())
    plot_g(dd,'Historical Drawdown')

    cagr=estimation.mean_historical_return(hist)
    std=hist.pct_change().dropna().std()*(252**0.5)
    sharpe=cagr/std
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('CAGR',str("{:.2%}".format(cagr)))
    col2.metric('STD',str("{:.2%}".format(std)))
    col3.metric('Sharpe ratio',str("{:.2f}".format(sharpe)))
    col4.metric('Max Drawdown',str("{:.2%}".format(dd.min())))

    ##############################
    ## HIST MAX LOSS AFTER X DAYS
    ##############################
    st.subheader('Minimum days to avoid losses')

    # create data
    a=[]
    for i in range(100, int(len(hist)/2)):
        a.append((((hist.shift(-i)/hist).dropna()-1).cummin().min()))
    a=pd.DataFrame(a)

    # write min. days
    if len(a[a<0].dropna())>= int(len(hist)/2)-100:
        st.metric('Min. days to avoid losses', '+∞')
    else:
        st.metric('Min. days to avoid losses', len(a[a<0].dropna()))

    # Plot
    fig = plt.figure()
    plt.plot(a,color='black')
    plt.axhline(0)
    plt.title('Max. loss after x days - Historical analysis')
    st.pyplot(fig)




    ##############################
    ## MONTE CARLO
    ##############################
    st.header('Monte Carlo Simulation')

    #Input
    n_mc=st.slider('Define the number of simulations: ',min_value=100,max_value=1000)
    n_t=st.slider('Define the number of days for the simulations: ',min_value=20,max_value=250)
    
    # Mean and cov computation
    mu_df=estimation.mean_historical_return(data,frequency=1)
    meanM = np.full(shape=(n_t, len(weights)), fill_value=mu_df).T
    portf_returns = np.full((n_t,n_mc),0.)
    try:
        cov=estimation.CovMatrix(rets)
        L = np.linalg.cholesky(cov)
    except:
        cov=rets.cov()
        L = np.linalg.cholesky(cov)

    for i in range(0,n_mc):
        Z = np.random.standard_t(12,size=(n_t, len(weights)))#uncorrelated RV's
        dailyReturns = meanM + np.inner(L,Z) #Correlated daily returns for individual stocks
        portf_r = np.cumprod(np.inner(weights,np.transpose(dailyReturns)) + 1)
        future_dates = [rets.index[-1] + timedelta(days=x) for x in range(0,n_t+1)]
        portf_returns[:,i] = portf_r


    ##############################
    ## PLOTTING AUX. FUNCTIONS
    ##############################

    def trend(r,n_t=n_t):
        L= np.zeros([n_t + 1, 1])
        L[0]=1
        for t in range(1, int(n_t)+1):
            L[t] = L[t-1]*(r**(1/n_t))
            L_df=pd.DataFrame(L,index=future_dates)
        return L_df

    ##############################
    ## PLOT MONTE CARLO SIM.
    ##############################

    #insert portfolio's start price
    ptf_returns=np.insert(portf_returns, 0, 1, axis=0)

    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(10)

    # Plot mc simulations
    with sns.color_palette("winter"):
        plt.plot(hist[-1]*pd.DataFrame(ptf_returns,index=future_dates))

    # Plot hist cum returns
    plt.plot(hist['2022':])

    # Plot trend lines
    plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.99)),color='black', label='99% probability')
    plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.01)),color='black',)
    plt.plot(hist[-1]*trend(np.quantile(portf_returns[-1],q=0.5)),color='white')

    # Display the chart
    st.pyplot(f)

    # Write some stats
    st.write('Forecasted returns')
    col1, col2, col3 = st.columns(3)
    col1.metric('Maximum loss - Probability 99%',str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.01)-1))))
    col2.metric('Maximum profit - Probability 99%', str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.99)-1))))
    col3.metric('Average return after '+str(n_t)+' days',str("{:.2%}".format((np.quantile(portf_returns[-1],q=0.5)-1))))

    col1, col2= st.columns(2)
    col1.metric('Worst realization',str("{:.2%}".format(portf_returns.min()-1)))
    col2.metric('Best realization',str("{:.2%}".format(portf_returns.max()-1)))





