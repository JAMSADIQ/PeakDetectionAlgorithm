# Tom Dent and Jam Sadiq
#This code  given a set of data will make a KDE and produce most prominent peak
#python PeakAlgorithm.py   --deltaVal 1 2 3 4 5 6 7 8 --input-data BBHpublic_O3a_medianmass_1_source.hdf  

from utils_awkde import *
import h5py
import argparse
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description=__doc__)
deltaChoices = [1,2,3,4,5,6,7,8] #[5,10,15...]]
parser.add_argument('--deltaVal', default=deltaChoices, nargs='+', type=float, help='width of window to measure peak height')
bwchoices = np.logspace(-1, -0.3, 15).tolist() #maxb=0.5
parser.add_argument('--bw-grid', type=float, default= bwchoices, nargs='+', help='grid of choices of global bandwidth, max bw can be 1 for that use bwchoices = np.logspace(-1, -0.1, 15).tolist()')
alphachoices = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
parser.add_argument('--alpha-grid', nargs="+", default=alphachoices, type=float, help='grid of choices of sensitivity parameter alpha for local bandwidth')
parser.add_argument('--input-data', type=str, help='Path to HDF file with data in a group from with keys name file000x. modify code for different type of data. see commenetd line for a one column txt data')
parser.add_argument('--pathdata', default=os.getcwd(), type=str, help='Path where data will be saved, default is current directory')
parser.add_argument('--optimalbw-criteria', type=str, default='max_height', help='optimize bandwidthusing "max_height"  of peak or "ratio" of peak_height__by__95th_percentile_value.')
opts = parser.parse_args()


#Analytic function for weigthing KDE to handle steep power law: 
def analytic_func(x, *data):
    """
    inputs:
       x = power in power law m^x 
       data = (min, max, data_array) 
         must be 3 entries with min, max of data_array and data_array 
    outputs:
       return analytic function to use in scipy.optimize to find root x
     
    analytic expression to be used in 
    KDE computation to deal with 
    steep power law
    idea:
     p(m) = C m^-Y    (gamma)
     C  = integral m^a from mmin= min(data_array) to mmax=max(data_array) d_data
     logLikelihood = NlnC +  a * sum_[i=1...N] ln(data_array)
     max logLikelihood solving d(loglikelihood)/da = 0 to get
    """
    min_D, max_D, data_array = data
    min_D, max_D, data_array = data
    OnePlusGamma = 1 + x
    return np.mean(np.log(data_array)) + 1 / OnePlusGamma - (max_D**(OnePlusGamma) * np.log(max_D) - min_D**(OnePlusGamma) * np.log(min_D)) / (max_D**OnePlusGamma - min_D**OnePlusGamma)


# Peak dectection algorithm
def find_peak_height(x_eval, kde_val, sigma_error_array_val, delta_val=5.0, bandwidth_val=0.5, showplot=False):
    """
    x_eval = test_data on which kde is computed
    kde_val = resulting KDE values on specified vlaues of input parameter [3-100] Msun
    sigma_error_array_val = error array are 95th percentile results of kde
    kwargs:
       delta_val = width of window to determine peak height:
          default = 5 but use fixed or factor of sigma fromawkDE code
       showplot = to plot KDE with all and most prominent peak
       bandwidth_val =  to use bandwidth choice for KDE, alpha = 1 by default
    output:
    max of peak heights, 
    highest peak location
    error at highest peak
       
    algorithm:
        use scipy to get local maxima as peaks 
        and indexes of peaks for kde vals
        now compute peak heights using 
        peak_height = average of vertical distance \
                      of kde values at peak and at + \
                       and - delta distance from the peak
       get max of thes epeak heights and get associated location
       and error at height from awKDE coeff
       Note: 
        if peak +- delta distance 
        goes outside kde we call it an edge case and use
        left or right vertical height for these edge cases.
    """
    #peak indexes
    peaks, _ = find_peaks(kde_val)
    peak_heights = []
    for i in range(len(peaks)):
        #find delta distance on left and right of each peak
        x_plus_delta = x_eval[peaks[i]] + delta_val
        x_minus_delta = x_eval[peaks[i]] - delta_val
        index_plus = np.searchsorted(x_eval, x_plus_delta)
        index_minus = np.searchsorted(x_eval, x_minus_delta)
        y_at_peak = kde_val[peaks[i]]
        y_at_minus_delta = kde_val[index_minus]
        y_at_plus_delta =  kde_val[index_plus-1]
        peak_height_val = 0.5 * ((y_at_peak - y_at_plus_delta) + (y_at_peak - y_at_minus_delta))
        peak_heights.append(peak_height_val)
    #max_peak_height
    peak_array = np.array(peak_heights)
    peak_loc_array = x_eval[peaks]
    sigma_error_at_peaks = sigma_error_array_val[peaks] 
    if len(peak_array) == 0:
        maxpeak, maxpeakloc, error = 0, kde_val[0], 1
    else:
        indx= np.argmax(peak_array)
        maxpeak, maxpeakloc, error = peak_array[indx], peak_loc_array[indx], sigma_error_at_peaks[indx]
    if showplot==True:
        plt.figure()
        plt.plot(x_eval, kde_val, label=r'$awKDE \alpha=1, bw={0:.2f}, \delta={1:.2f}$'.format(bandwidth_val,  delta_val)) 
        plt.plot(x_eval[peaks], kde_val[peaks],"kx")
        plt.xlabel("M")
        plt.ylabel("p(M)")
        plt.legend()
        for i in range(len(peaks)):
            plt.errorbar(x_eval[peaks[i]], kde_val[peaks[i]], yerr=peak_heights[i], uplims=True, color='k', )
        plt.errorbar(x_eval[peaks[np.argmax(peak_heights)]], kde_val[peaks[np.argmax(peak_heights)]], yerr=max(peak_heights), uplims=True, color='b', )
        #plt.savefig('Peak_heights_in_kde_with_bw_{0}_delta_{1}.png'.format(bandwidth_val, delta_val))
        #plt.close()
        plt.show()
    return maxpeak, maxpeakloc, error  


#we have a criteria for optbw :max height or ratio for all bw choices
def get_optbw_peakstuff_for_allbw_using_det_stat(train_data, test_data, gamma_power, bwgrid, delta, method='global_sigma_only', optbw_criteria='max_height', compare_kde_plot=False):
    """
    inputs:
    train_data
    test_data = we will generate data in main function
    bwgrid  = choices of bandwidths 
    delta  = either fixed delta or factor for global sigma
    method = for opt bw we get peaks or ratio via 
m utils_awkde import *
         'global_sigma_only', or 'global_sigma_factors' or 'fixed_delta'
   optbw_criteria = 'max_height' or 'ratio'
         after we choose method we have two case for maximum peak
         its height
         ratio
    output:
    return opt bw, globalsigma, peak height, peak location, ratio for best bw choice
    """
    height_list = []
    location_list = []
    global_sigma_list = []
    error_list = []
    ratio_list = []
    for k in range(len(bwgrid)):
        kde_val = bounded_kde_awkde(train_data, test_data, alp=1.0, gl_bandwidth=bwgrid[k], ret_kde=False, xlow=min(test_data), xhigh=max(test_data))
        error_array = bounded_get_sigma_error(train_data, test_data, bwgrid[k], 1.0, xlow=min(test_data), xhigh=max(test_data), pseudo=True)
        new_kde_val =  kde_val * test_data**(-gamma_power)
        kde_ratio = kde_val/new_kde_val
        normfactor = max(kde_ratio) 
        if compare_kde_plot==True:
            plt.figure()
            plt.plot(test_data, kde_val , 'k--', linewidth=2, label='std kde')
            plt.plot(test_data, new_kde_val * normfactor, 'r-.', linewidth=1, label='factored kde')
            plt.legend()
            plt.savefig('standard_vs_weighted_KDE_with_with_{0:.2f}_bw{1:.2f}.png'.format(gamma_power, bwgrid[k]))
            plt.close()
        new_error_array = normfactor * (error_array * test_data**(-gamma_power))
        globalsigma = get_global_sigma_value_from_KDE_variance(train_data, test_data, bwgrid[k], 1.0)
        if method=='global_sigma_only':
            peak_height, peak_location, peak_error = find_peak_height(test_data, new_kde_val, error_array, delta_val=globalsigma * 1.0, bandwidth_val=bwgrid[k], showplot=False)
        elif  method=='global_sigma_factor':
            peak_height, peak_location, peak_error = find_peak_height(test_data, new_kde_val, new_error_array, delta_val=globalsigma * delta, bandwidth_val=bwgrid[k], showplot=False)
        else:
            print("method is fixed_delta")
            peak_height, peak_location, peak_error = find_peak_height(test_data, new_kde_val, error_array, delta_val=delta, bandwidth_val=bwgrid[k], showplot=False)
        height_list.append(peak_height)
        location_list.append(peak_location)
        error_list.append(peak_error)
        global_sigma_list.append(globalsigma)    #may want to save
        ratio_list.append(peak_height / peak_error)  #this for pseudo error removing term

    if optbw_criteria == 'max_height':
        max_indx = np.argmax(height_list)
    else:
        max_indx = np.argmax(ratio_list)
    print('optbw, height, location = ', bwgrid[max_indx], height_list[max_indx], location_list[max_indx])
    return bwgrid[max_indx], global_sigma_list[max_indx], height_list[max_indx], location_list[max_indx], ratio_list[max_indx], error_list[max_indx]


def experiment_allbwds(dataf, bwgrid, alphagrid, delta_lists, eval_points=200, methodchoice='global_sigma_only', initial_optbw_criteria='max_height'):
    """
    inputs:
     dataf: file containing mock data 
     bwgrid: choice pf bw for kde
     alphagrid: choice of alpha for kde
     delta_lists: fixed delta choices
     eval_points: total data-points in kde
     methodchoice=  global_sigma_only, global_sigma_factor , fixed delta
     initial_optbw_criteria='max_height', ratio
    return:  
      optimum bw lists for all mock datasets for each fixed delta
      global sigma  from code for each delta
      peak height
      peak locations

    """
    opt_bw_list = [[] for _ in range(len(delta_lists))]
    peaks_height_list = [[] for _ in range(len(delta_lists))]
    peaks_location_list = [[] for _ in range(len(delta_lists))]
    global_sigma_list = [[] for _ in range(len(delta_lists))]
    errorVal_list = [[] for _ in range(len(delta_lists))] #if needed change previous function
    ratio_peak_by_error_list = [[] for _ in range(len(delta_lists))]
    f = h5py.File(dataf, 'r')
    # here we assume specific hdf file but it can be txt file or an HDF file with different samples
    grp_data = f['data']
    for k in grp_data.keys():
        print('dataset_number = ', k)
        train_data =  grp_data[k][...]
        min_traindata, max_traindata = min(train_data), max(train_data)
        #weighting the KDE  
        gamma_pow = fsolve(analytic_func, [-1.5], (min_traindata, max_traindata,train_data))
        #print("gamma_power = ", gamma_pow[0])
        range_data = max_traindata - min_traindata #max(train_data) - min(train_data)
        minval = min_traindata - 0.01 * range_data # change 0.01 to 0.1 
        maxval = max_traindata + 0.01 * range_data
        test_data = np.linspace(minval, maxval, eval_points)
        for  j in range (len(delta_lists)):
            bwVal, global_sigmaVal, heightVal, locationVal, ratioVal, errorVal = get_optbw_peakstuff_for_allbw_using_det_stat(train_data, test_data, gamma_pow[0], bwgrid, delta_lists[j], method=methodchoice, optbw_criteria=initial_optbw_criteria)
            #Plots for optimized BWs for each delta
            kde_val = bounded_kde_awkde(train_data, test_data, alp=1.0, gl_bandwidth=bwVal, ret_kde=False, xlow=min(test_data), xhigh=max(test_data))
            error_array = bounded_get_sigma_error(train_data, test_data, bwVal, 1.0, xlow=min(test_data), xhigh=max(test_data), pseudo=True)
            new_kde_val =  kde_val * test_data**(-gamma_pow)
            peak_height, peak_location, peak_error = find_peak_height(test_data, new_kde_val, error_array, delta_val=global_sigmaVal*delta_lists[j] , bandwidth_val=bwVal, showplot=True)
            #save the date in list for each choice of delta
            opt_bw_list[j].append(bwVal)
            peaks_height_list[j].append(heightVal)
            peaks_location_list[j].append(locationVal)
            global_sigma_list[j].append(global_sigmaVal)
            ratio_peak_by_error_list[j].append(ratioVal)
            errorVal_list[j].append(errorVal)
    return opt_bw_list, global_sigma_list, peaks_height_list, peaks_location_list, ratio_peak_by_error_list, errorVal_list

Alp = [1.0] #opts.alpha_grid  for more choices
BWs = opts.bw_grid
factors_sigma = opts.deltaVal #[1,2,3,4,5,6,7,8]
datafile =  opts.input_data
#generic global sigma case  with ratio criteria for opt bw
experiment = 'global_sigma_factor'
criteria =  'ratio'

opt_bws, opt_global_sigma, opt_peak_height, opt_peak_location, opt_peak_height_by_error_ratio, opt_errorValues= experiment_allbwds(datafile, BWs, Alp, factors_sigma, eval_points=200, methodchoice='global_sigma_factor', initial_optbw_criteria='ratio')                            
print(opt_bws, opt_global_sigma, opt_peak_height, opt_peak_location)
