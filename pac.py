import pandas as pd
import numpy as np
import yfinance as yf

def dollarcost(tickers,weights,invest,day=15):

    data=yf.download(tickers)['Adj Close'].dropna()
    returns=data.pct_change().dropna()
    data=data.iloc[1:]

    dates=returns[data.index.day==day].index
    dates_=[]
    for x in dates: dates_.append(returns.index.get_loc(x))

    wealth=np.zeros(len(returns))

    if dates[-1]!= returns.index[-1]:
        dates_.append(len(returns))

    c=0
    x=0

    invested=np.zeros(len(returns))
    ptf_r=np.zeros((len(returns)))
    cash=np.zeros(len(returns))
    rets=np.zeros(len(dates_))
    
    for i in dates_:
        i=i-1

        w1=((np.array(weights)*invest)/data.iloc[c]).astype(int)*data.iloc[c]/invest
        ptf_r[c:i]=returns[c:i]@np.array(w1)
        cash[c:i]=np.ones((i-c))*invest-(w1*invest).sum()
        inv=invest+cash[c]

        if c==0:
            wealth[c]=inv -cash[c]
        else:
            invested[c:i]=invested[c-1]+inv-cash[c]
            
            wealth[c]=wealth[c-1]*(1+ptf_r[c])+inv-cash[c]

        wealth[c+1:i]=wealth[c]*np.cumprod((1+ptf_r[c+1:i]))
        rets[x]=wealth[i-1]/wealth[c]

        c=i
        x=x+1

        wealth1=wealth[wealth!=0]
        cash1=cash[cash!=0]
        invested1=invested[invested!=0]

    we=pd.DataFrame(wealth1,index=returns.index[range(0,len(wealth1))],columns=['Wealth index'])
    c=pd.DataFrame(cash1,index=returns.index[range(0,len(cash1))],columns=['Wealth index'])
    invested=pd.DataFrame(invested1,index=returns.index[range(0,len(invested1))],columns=['Wealth index'])
    mwrr=rets[rets!=0].prod()**(1/len(rets[rets!=0]))-1

    return we, c,invested, mwrr,rets
