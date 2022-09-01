from ast import Pass
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import riskfolio as rp
import datetime
import estimation



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

    st.title('Vanilla Risk Parity Optimization')
    #st.markdown('Based on Gambeta and Kwon. Risk return trade-off in relaxed risk parity portfolio optimization. Journal of Risk and Financial Management, 2020. https://doi.org/10.3390/jrfm13100237')

    n_stocks=st.slider('Select the number of assets: ',min_value=2,max_value=12)
    n_loockback=st.slider('Select the minium days for the portfolio optimization: ',min_value=250,max_value=500)
    n_rebal=st.slider('Define the rebalancing frequency (every x months)',min_value=6,max_value=12)
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

    if st.button('Run optimization'):
        try:
            ##############################
            # RETURNS COMPUTATION & PLOTTING
            ##############################
            rets=data.pct_change().dropna()
            cumrets=(rets+1).cumprod()

            ##############################
            ## REBALANCING DATES
            ##############################
            # Selecting last day of month of available data
            index = rets.groupby([rets.index.year, rets.index.month]).tail(1).index
            index_2 = rets.index
            # Quarterly Dates
            index = [x for x in index if float(x.month) % n_rebal== 0 ] 

            # Dates where the strategy will be backtested
            index_ = [index_2.get_loc(x) for x in index if index_2.get_loc(x) > n_loockback]

            weights = pd.DataFrame([])
            ##############################
            ## OPTIMIZATION
            ##############################
            for i in index_:
                    Y = rets.iloc[:i,:] 

                    # Building the portfolio object
                    port = rp.Portfolio(returns=Y)
                    
                    # Estimate optimal portfolio:
                    port.assets_stats(method_mu='hist', method_cov='fixed')

                    w = port.rp_optimization(model='Classic', rm='MV' )

                    if w is None:
                        w = weights.tail(1).T
                    weights = pd.concat([weights, w.T], axis = 0)

            weights['index']=index[-len(weights.index):]
            weights.set_index(weights['index'],inplace=True)
            weights.drop('index',axis=1,inplace=True)
            
            

            ax  = weights.plot.bar(stacked=True)
            fig = ax.get_figure()
            fig.patch.set_facecolor('#00172B')
            ax.patch.set_facecolor('#00172B')
            plt.xticks(color='white')
            plt.title('Portfolio weights',color='white')
            plt.yticks(color='white')
            plt.ylabel('')
            st.pyplot(fig)

            index_.append(len(rets)-1)
            first=True
            a=pd.Series()
            c=0
            x=0
            for i in index_:
                if first==True:
                    first=False
                else:
                    a=pd.concat([(rets.iloc[x:i]@np.array(weights.iloc[c])),a],join='inner')
                    c+=1
                x=i

            a.sort_index(inplace=True)

            hist=(a+1).cumprod()
            plot_g(hist,'Portfolio Historical Value')

            #DRAWDOWN
            running_max = np.maximum.accumulate(hist)
            dd = hist / running_max - 1 
            plot_g(dd,title='Underwater Plot')

            
            st.subheader('Historical Stats')
            cagr=estimation.mean_historical_return(hist)
            std=hist.pct_change().dropna().std()*(252**0.5)
            sharpe=cagr/std
            col1, col2, col3, col4 = st.columns(4)
            col1.metric('CAGR',str("{:.2%}".format(cagr)))
            col2.metric('STD',str("{:.2%}".format(std)))
            col3.metric('Sharpe ratio',str("{:.2f}".format(sharpe)))
            col4.metric('Max Drawdown',str("{:.2%}".format(dd.min())))

            st.subheader('Worst five drawdowns')
            st.write(dd_table(dd,hist))
        except:
            st.write('You need to define the inputs')


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
if __name__ == "__main__":
    main()
