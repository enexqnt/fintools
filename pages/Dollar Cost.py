from ast import Pass
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import estimation
import scipy
import pac



st.set_page_config('FinTools')
##############################
# PLOT FUNCTION
##############################
def plot_g(data,title=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor('#00172B')
    ax.patch.set_facecolor('#00172B')
    plt.plot(data)
    plt.title(title,color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    st.pyplot(fig)


def get_max_drawdown_underwater(underwater):
    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def dd_table(underwater,hist):
    drawdowns = []
    for _ in range(5):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if ((len(hist) == 0)
                or (len(underwater) == 0)
                or (np.min(underwater) == 0)):
            break
    df_drawdowns = pd.DataFrame(index=list(range(5)),
                                columns=['Net drawdown in %',
                                         'Peak date',
                                         'Valley date',
                                         'Recovery date',
                                         'Duration'])
    for i, (peak, valley, recovery) in enumerate(drawdowns):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'Duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                                .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'Recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Net drawdown in %'] = (
            (hist.loc[peak] - hist.loc[valley]) / hist.loc[peak]) * 100

    df_drawdowns['Peak date'] = pd.to_datetime(df_drawdowns['Peak date'])
    df_drawdowns['Valley date'] = pd.to_datetime(df_drawdowns['Valley date'])
    df_drawdowns['Recovery date'] = pd.to_datetime(
        df_drawdowns['Recovery date'])
    return df_drawdowns

def main():
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    st.title('Dollar Cost Averaging (PAC)')

    n_stocks=st.slider('Select the number of assets: ',min_value=2,max_value=12)
    invv=st.slider('Define how much do you want to invest each month: ',min_value=100,max_value=10000)

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
        hist,c,i,mw,rr=pac.dollarcost(tickers,weights,1000,15)
        plot_g(hist,'Historical cumulative returns')

        ##############################
        ## DRAWDOWN & other metrics
        ##############################
        running_max = np.maximum.accumulate(hist)
        dd = hist / running_max - 1 
        plot_g(dd,'Historical Drawdown')

        cagr=mw
        std=rr.std()
        sharpe=cagr/std
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('CAGR',str("{:.2%}".format(cagr)))
        col2.metric('STD',str("{:.2%}".format(std)))
        col3.metric('Sharpe ratio',str("{:.2f}".format(sharpe)))
        col4.metric('Max Drawdown',str("{:.2%}".format(float(dd.min()))))

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
        plot_g(a,'Max. loss after x days - Historical analysis')
        ##############################
        ## COVARIANCE MATRIX
        ##############################
        st.header('Covariance estimation')
        try:
            cov=estimation.CovMatrix(rets)
            L = np.linalg.cholesky(cov)
            st.metric('Covariance Estimation Method:','Robust')
            st.write('*Denoising and Detoning by De Prado')
        except:
            cov=rets.cov()
            L = np.linalg.cholesky(cov)
            st.metric('Covariance Estimation Method:','Classical')
            st.write('*No Robut Covariance Estimation adopted since the var/cov matrix is not semi-positive definite')
        
        fig = plt.figure()
        sns.heatmap(cov*np.sqrt(252), cmap="winter", annot=False)
        plt.title('Variance/Covariance matrix')
        st.pyplot(fig)
        plt.show()



if __name__ == "__main__":
    main()
