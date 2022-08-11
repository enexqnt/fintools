import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity


def returns_from_prices(prices, log_returns=False):
    if log_returns:
        returns = np.log(1 + prices.pct_change()).dropna(how="all")
    else:
        returns = prices.pct_change().dropna(how="all")
    return returns

def prices_from_returns(returns, log_returns=False):
    if log_returns:
        ret = np.exp(returns)
    else:
        ret = 1 + returns
    ret.iloc[0] = 1  # set first day pseudo-price
    return ret.cumprod()


def mean_historical_return(
    prices, returns_data=False, compounding=True, frequency=252, log_returns=False
):
    if returns_data:
        returns = prices.dropna()
    else:
        returns = returns_from_prices(prices, log_returns).dropna()

    if compounding:
        return (1 + returns).prod() ** (frequency / returns.count()) - 1
    else:
        return returns.mean() * frequency



def CovMatrix(X):
        cov = np.cov(X.T)
        T, N = X.shape
        q = T / N
        cov = denoiseCov(cov, q)
        return cov
def cov2corr(cov):
    cov1 = np.array(cov, ndmin=2)
    std = np.sqrt(np.diag(cov1))
    corr = np.clip(cov1 / np.outer(std, std), a_min=-1.0, a_max=1.0)

    return corr

def corr2cov(corr, std):
    flag = False
    if isinstance(corr, pd.DataFrame):
        cols = corr.columns.tolist()
        flag = True

    cov = corr * np.outer(std, std)

    if flag:
        cov = pd.DataFrame(cov, index=cols, columns=cols)

    return cov

def denoiseCov(cov, q,bWidth=0.01, detone=True, mkt_comp=1, alpha=0):
    flag = False
    if isinstance(cov, pd.DataFrame):
        cols = cov.columns.tolist()
        flag = True

    corr = cov2corr(cov)
    std = np.diag(cov) ** 0.5
    eVal, eVec = getPCA(corr)
    eMax, var = findMaxEval(np.diag(eVal), q, bWidth)
    nFacts = eVal.shape[0] - np.diag(eVal)[::-1].searchsorted(eMax)
    
    corr = denoisedCorr(eVal, eVec, nFacts)

    if detone == True:
        eVal_ = eVal[:mkt_comp, :mkt_comp]
        eVec_ = eVec[:, :mkt_comp]
        corr_ = np.dot(eVec_, eVal_).dot(eVec_.T)
        corr = corr - corr_

    cov_ = corr2cov(corr, std)

    if flag:
        cov_ = pd.DataFrame(cov_, index=cols, columns=cols)

    return cov_

def getPCA(matrix):
    # Get eVal,eVec from a Hermitian matrix
    eVal, eVec = np.linalg.eigh(matrix)
    indices = eVal.argsort()[::-1]  # arguments for sorting eVal desc
    eVal, eVec = eVal[indices], eVec[:, indices]
    eVal = np.diagflat(eVal)

    return eVal, eVec


def findMaxEval(eVal, q, bWidth=0.01):

    out = minimize(
        lambda *x: errPDFs(*x), 0.5, args=(eVal, q, bWidth), bounds=((1e-5, 1 - 1e-5),)
    )

    if out["success"]:
        var = out["x"][0]
    else:
        var = 1

    eMax = var * (1 + (1.0 / q) ** 0.5) ** 2

    return eMax, var

def errPDFs(var, eVal, q, bWidth=0.01, pts=1000):
    # Fit error
    pdf0 = mpPDF(var, q, pts)  # theoretical pdf
    pdf1 = fitKDE(eVal, bWidth, x=pdf0.index.values)  # empirical pdf
    sse = np.sum((pdf1 - pdf0) ** 2)

    return sse

def mpPDF(var, q, pts):
    if isinstance(var, np.ndarray):
        if var.shape == (1,):
            var = var[0]

    eMin, eMax = var * (1 - (1.0 / q) ** 0.5) ** 2, var * (1 + (1.0 / q) ** 0.5) ** 2
    eVal = np.linspace(eMin, eMax, pts)
    pdf = q / (2 * np.pi * var * eVal) * ((eMax - eVal) * (eVal - eMin)) ** 0.5
    pdf = pd.Series(pdf, index=eVal)

    return pdf

def fitKDE(obs, bWidth=0.01, kernel="gaussian", x=None):
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, 1)

    kde = KernelDensity(kernel=kernel, bandwidth=bWidth).fit(obs)

    if x is None:
        x = np.unique(obs).reshape(-1, 1)

    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    logProb = kde.score_samples(x)  # log(density)
    pdf = pd.Series(np.exp(logProb), index=x.flatten())

    return pdf

def denoisedCorr(eVal, eVec, nFacts, kind="fixed"):
    eVal_ = np.diag(eVal).copy()

    if kind == "fixed":
        eVal_[nFacts:] = eVal_[nFacts:].sum() / float(eVal_.shape[0] - nFacts)
    elif kind == "spectral":
        eVal_[nFacts:] = 0

    eVal_ = np.diag(eVal_)
    corr = np.dot(eVec, eVal_).dot(eVec.T)
    corr = cov2corr(corr)

    return corr