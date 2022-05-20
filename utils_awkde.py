"""
This is a module for functions related to awkde code
"""
from awkde import GaussianKDE
import numpy as np
import operator
import scipy.stats as st


def kde_awkde(x, x_grid, alp=0.5, gl_bandwidth='silverman', ret_kde=False):
    """Kernel Density Estimation with awkde 
    inputs:
      x = training data 
      x_grid = testing data
      alp = smoothing factor for local bw
      gl_bandwidth = global bw for kde
      kwargs:
       ret_kde optional 
        if True kde will be output with estimated kde-values 
    """
    kde = GaussianKDE(glob_bw=gl_bandwidth, alpha=alp, diag_cov=True)
    kde.fit(x[:, np.newaxis])
    if isinstance(x_grid, (list, tuple, np.ndarray)) == False:
        y = kde.predict(x_grid)
    else:
        y = kde.predict(x_grid[:, np.newaxis])

    if ret_kde == True:
        return kde, y
    return y


def bounded_kde_awkde(x, x_grid, alp=0.5, gl_bandwidth='silverman', ret_kde=False, xlow=None, xhigh=None):
    """ Bounded Kernel Density Estimation with awkde 
    inputs:
      x = training data 
      x_grid = testing data
      alp = smoothing factor for local bw
      gl_bandwidth = global bw for kde
      kwargs:
       ret_kde optional 
        if True kde will be output with estimated kde-values 
       xlow=None, xhigh=None  
        for bounds at the edges of kde   
    """
    kde = GaussianKDE(glob_bw=gl_bandwidth, alpha=alp, diag_cov=True)
    kde.fit(x[:, np.newaxis])
    if isinstance(x_grid, (list, tuple, np.ndarray)) == False:
        y = kde.predict(x_grid)
    else:
        y = kde.predict(x_grid[:, np.newaxis])
    if xlow is not None:
        newx_grid = 2*xlow - x_grid
        y += kde.predict(newx_grid[:, np.newaxis]) 
    if xhigh is not None:
        newx_grid = 2*xhigh - x_grid
        y += kde.predict(newx_grid[:, np.newaxis])  
    if ret_kde == True:
        return kde, y
    return y


def two_fold_crossvalidation(sample, bwchoice, alphachoice, n=5):
    """
    inputs:
      samples: training set
      bwchoice: choices of bwd for kde
      alphachoice: choice of kde parameter [0.1, 1]

    shuffle data into two equal samples
    train in half and test on other half
    in kde. 
    try this 5 times and take average 
    of those np.mean(combined 5 results)
    fom = np.log(average)
    """
    #randomly select data    
    random_data = np.random.choice(sample, len(sample))
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice)
    fomlist =[]
    for i in range(int(n)):
        #split data into two subset of equal size [or almost equal (n+1, n) parts]
        x_train, x_eval = np.array_split(random_data, 2)
        y = kde_awkde(x_train, x_eval, alp=alphachoice, gl_bandwidth=bwchoice)
        fomlist.append(np.log(y))
    return np.mean(fomlist)


def get_optimized_bw_kde_using_twofold(traindata, testdata, bwgrid, alphagrid):
    """ 
    Given grid of bw_choices 
    use two-fold cross_validation to get opt_bw
    figure of merit from two_fold_cv function for each choice
    return optimal param: optbw , optalpha
    """
    FOM= {}
    for gbw in bwgrid:
        for alphavals in alphagrid:
            FOM[(gbw, alphavals)] = two_fold_crossvalidation(traindata, gbw, alphavals)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    #get kde using optimized alpha and optimized bw
    kde_result = kde_awkde(traindata, testdata, alp=optalpha, gl_bandwidth=optbw)
    return optbw, optalpha, kde_result


def loocv_awkde(sample, bwchoice, alphachoice):
    """
    Use specific choice of alpha and bw 
    we use Leave one out cross validation
    on Awkde kde fit
    LOOCV:
    we train n-1 of the n sample leaving one
    in n iterations and compute kde fit on 
    n-1 samples. For each iteration we  use this kde 
    to predict the missed ith sample of ith iteration.
    We take log of this predicted value and
    sum these all values (N values if len Sample = N)
    We output this sum.
    fom  = log(predict(miss value)) is called 
    a figure of merit.
    """
    if bwchoice not in ['silverman', 'scott']:
        bwchoice = float(bwchoice) #bwchoice.astype(np.float) #for list
    fom = 0.0
    for i in range(len(sample)):
        leave_one_sample, miss_sample = np.delete(sample, i), sample[i]
        y = kde_awkde(leave_one_sample, miss_sample, alp=alphachoice, gl_bandwidth=bwchoice)
        fom += np.log(y)
    return fom


def get_optimized_bw_kde_using_loocv(sample, alphagrid, bwgrid):
    """ 
    Given grid of alpha and bw choice it will get 
    figure of merit from loocv_awkde function for each choice
    return a dictionary of FOM:  FOM[(bw, alpha)]
    and  optimal param and FOM at these params:  optbw , optalpha, FOM[(optbw , optalpha)] 
    """

    FOM= {}
    for gbw in bwgrid:
        for alphavals in alphagrid:
            FOM[(gbw, alphavals)] = loocv_awkde(sample, gbw, alphavals)
    optval = max(FOM.items(), key=operator.itemgetter(1))[0]
    optbw, optalpha  = optval[0], optval[1]
    return  optbw, optalpha


def get_global_sigma_value_from_KDE_variance(traindata, testdata, optbw, optalpha, err=False):
    """
    get global sigma from kde code 
    its the sigma in Gaussian Kernel taken from 1/sqrt(2pisigma^2)
    #see https://github.com/JAMSADIQ/awkde/blob/master/awkde/awkde.py
    and  
    #https://github.com/JAMSADIQ/awkde/blob/master/cpp/backend.cpp 
    inputs:
      traindata  = array for which to prepare a kde 
      testdata   = array of resulting kde [only needed if we want 905 confidence curves]
      optbw   = bw used in kde 
      optalpha = alpha used in kde 
    kwargs: 
      err=False by deafault
      if True it will compute 90% conficence curve 
    output:
      if err =False [bydeafult] return only global sigma value
      if err= True returns gloabl sigma value and an array of upper  90 percentile  curve [95% interval]
    """
    kde2 = GaussianKDE(glob_bw=optbw, alpha=optalpha, diag_cov=True)
    kde2.fit(traindata[:, np.newaxis])
    global_sigma_val = kde2.get_global_sigma(testdata[:, np.newaxis])  

    if err == True:
        kde_result = kde_awkde(traindata, testdata, alp=optalpha, gl_bandwidth=optbw)
        coeff = kde2.predict2(testdata[:, np.newaxis])
        sqcoeff = coeff**2
        cjmean = np.zeros(len(kde_result))
        cjsqmean =  np.zeros(len(kde_result))
        for k in range(len(kde_result)):
            cjmean[k] = sum(coeff[k])/len(coeff[k])
            cjsqmean[k] = sum(sqcoeff[k])/len(sqcoeff[k])
        sigmaarray = np.sqrt(cjsqmean - cjmean**2)
        PN = st.norm.ppf(.95) #1.644
        Ndetection = len(traindata)
        error_estimate =  kde_result + (PN) * sigmaarray / np.sqrt(Ndetection)
        return global_sigma_val, error_estimate
    else:
        return global_sigma_val


#define  A NEW FUNCTION TO AVOID GLOBAL SIMA PLUS extra kde again not object but as output
def get_sigma_error(kde_result, traindata, testdata, optbw, optalpha):
    """
    get delta from kde code 
    its the sigma in Gaussian Kernel taken from 1/sqrt(2pisigma^2)
    #see https://github.com/JAMSADIQ/awkde/blob/master/awkde/awkde.py
    and  
    #https://github.com/JAMSADIQ/awkde/blob/master/cpp/backend.cpp 
    """
    kde2 = GaussianKDE(glob_bw=optbw, alpha=optalpha, diag_cov=True)
    kde2.fit(traindata[:, np.newaxis])
    coeff = kde2.predict2(testdata[:, np.newaxis])
    sqcoeff = coeff**2
    cjmean = np.zeros(len(kde_result))
    cjsqmean =  np.zeros(len(kde_result))
    for k in range(len(kde_result)):
        cjmean[k] = sum(coeff[k])/len(coeff[k])
        cjsqmean[k] = sum(sqcoeff[k])/len(sqcoeff[k])
    #sigmaarray = np.sqrt(cjsqmean - cjmean**2)
    #pseudo
    sigmaarray = np.sqrt(cjsqmean)# - cjmean**2)
    PN = st.norm.ppf(.95) #1.28155156554466004
    Ndetection = len(traindata)
    error_estimate =  kde_result + (PN) * sigmaarray / np.sqrt(Ndetection)
    nerror_estimate =  kde_result - (PN) * sigmaarray / np.sqrt(Ndetection)
    return error_estimate #, nerror_estimate

def bounded_get_sigma_error(traindata, testdata, optbw, optalpha, xlow=None, xhigh=None, pseudo=False):
    """
    get delta from kde code 
    its the sigma in Gaussian Kernel taken from 1/sqrt(2pisigma^2)
    #see https://github.com/JAMSADIQ/awkde/blob/master/awkde/awkde.py
    and  
    #https://github.com/JAMSADIQ/awkde/blob/master/cpp/backend.cpp 
    """
    kde2 = GaussianKDE(glob_bw=optbw, alpha=optalpha, diag_cov=True)
    kde2.fit(traindata[:, np.newaxis])
    coeff = kde2.predict2(testdata[:, np.newaxis])
    if xlow is not None:
        newtestdata = 2*xlow - testdata
        coeff += kde2.predict2(newtestdata[:, np.newaxis])
    if xhigh is not None:
        newtestdata = 2*xhigh - testdata
        coeff += kde2.predict2(newtestdata[:, np.newaxis])
    #print("coeffs shape = ", coeff.shape, len(coeff))
    sqcoeff = coeff**2
    cjmean = np.zeros(len(testdata))
    cjsqmean =  np.zeros(len(testdata))
    for k in range(len(testdata)):
        cjmean[k] = sum(coeff[k])/len(coeff[k])
        cjsqmean[k] = sum(sqcoeff[k])/len(sqcoeff[k])

    if pseudo==True:
        sigmaarray = np.sqrt(cjsqmean)  #pseudo no second term
    else:
        sigmaarray = np.sqrt(cjsqmean - cjmean**2) 
    num_sigma = st.norm.ppf(.95) #1.28155156554466004
    Ndetection = len(traindata)
    error_estimate =  num_sigma * sigmaarray / np.sqrt(Ndetection)
    return error_estimate  #, coeff # onlt to investigate


########## kde with error and global sigma in one function:
