
# coding: utf-8

# In[2]:

import pandas as pd
import h5py
import numpy
import re
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import scipy.io as sio

WINDOW_SIZE = 3
POLY_ORDER = 1
STEADY_EPSILON = 2
MIN_STEADY = 10

# In[3]:

TRAIN_FILE = "train.h5"
TEST_FILE = "test.h5"
BASELINE_WIENER = "baseline_wiener.csv"
BASELINE_KALMAN = "baseline_kalman.csv"
EMG_FILTER_LAG = 2


# In[4]:

def compute_differentials(array, order, axis):
    # results = [savgol_filter(numpy.array(array).T, WINDOW_SIZE, POLY_ORDER).T]
    results = [array]
    for i in range(order):
        if results[-1].shape[0] > 20:
            results.append(numpy.diff(savgol_filter(results[-1].T, WINDOW_SIZE, POLY_ORDER).T, axis=axis))
        else:
            results.append(numpy.diff(results[-1], axis=axis))
    return results

def cut_steady(arr, dims):
    check = arr[:,:dims]
    n = arr.shape[0]
    i = 1
    while i < n:
        if numpy.any(numpy.abs(numpy.mean(check[:i,:], axis=0) - check[i,:]) > STEADY_EPSILON):
            break
        i += 1
    i=1
    j = -2
    while j >= -n:
        if numpy.any(numpy.abs(numpy.mean(check[j:,:], axis=0) - check[j,:]) > STEADY_EPSILON):
            break
        j -= 1
    return arr[i-1:j+1,:], i - 1, j + 1

def interpolate_steady(arr, dims):
    check = arr[:,:dims]
    n = arr.shape[0]
    diff = numpy.diff(check, axis=0)
    abs_diff = numpy.sum(numpy.abs(diff), axis = 1)
    jumps = numpy.flatnonzero(abs_diff > numpy.std(abs_diff) * 5)
    if jumps.size:
        jump_point = jumps[0] + 1
        j = jump_point - 1
        while j >= 0:
            if numpy.any(numpy.abs(numpy.mean(check[j:jump_point,:], axis=0) - check[j,:]) > STEADY_EPSILON):
                break
            j -= 1
        j += 1
        if jump_point > j:
            arr[j:jump_point, :] = arr[j, :] + numpy.vstack((numpy.arange(jump_point - j), numpy.arange(jump_point - j))).T * (arr[jump_point, :] - arr[j, :]) / (jump_point - j)
        # return arr[j:jump_point,:], j, jump_point
        return arr[:j,:], j
    return arr, arr.shape[0]


# In[5]:

def align_arrays(arrays_list):
    final_size = min(map(len, arrays_list))
    return [array[-final_size:] for array in arrays_list]


# In[6]:

def cut_arrays(arrays_list, head=None, tail=None):
    return [array[head:tail] for array in arrays_list]


# In[7]:

def transform_emg(dataset):
    """For a dataset[n_ticks, n_emg_channels] computes a transforamtion with
    added time-shifted channels dataset[n_ticks -  2*EMG_FILTER_LAG, 3*n_emg_channels] by
    by pasting the emg values for the past EMG_FILTER_LAG ticks as additional channels."""
    emg_channels = dataset.shape[1]
    emg_lag = numpy.zeros((dataset.shape[0] - 2*EMG_FILTER_LAG, emg_channels*3))
    for l in range(EMG_FILTER_LAG+1):
        emg_lag[:, l*emg_channels:(l+1)*emg_channels] = dataset[EMG_FILTER_LAG-1+l:-EMG_FILTER_LAG-1+l]
    return emg_lag


# In[8]:

class BaselineModel(object):
    def fit(self, emg_list, coordinates_and_diffs_list):
        transformed_emg_list = list(map(transform_emg, emg_list))
        emg_lag = coordinates_and_diffs_list[0].shape[0] - transformed_emg_list[0].shape[0]
        assert(emg_lag == EMG_FILTER_LAG)
        coordinates_and_diffs_uncut = numpy.vstack(cut_arrays(coordinates_and_diffs_list, head=emg_lag))
        coordinates_and_diffs_lag = numpy.vstack(cut_arrays(
                coordinates_and_diffs_list, head=emg_lag, tail=-1))
        coordinates_and_diffs_now = numpy.vstack(cut_arrays(
                coordinates_and_diffs_list, head=1 + emg_lag))
        diffs_uncut = coordinates_and_diffs_uncut[:, 2:]
        # diffs_lag = coordinates_and_diffs_lag[:, 2:]
        # diffs_now = coordinates_and_diffs_now[:, 2:]
        emg = numpy.vstack(transformed_emg_list)
        emg = numpy.hstack((numpy.ones((emg.shape[0], 1)), emg))
        # clf = linear_model.Lasso(alpha=0.05)
        # clf.fit(emg, diffs_uncut)
        self.W = numpy.linalg.pinv(emg).dot(diffs_uncut).T
        # self.clf = clf
        # self.W[numpy.abs(self.W)<10] = 0
        # model_coo_and_diff = clf.predict(emg)
        model_coo_and_diff = emg.dot(self.W.T)
        model_coo_and_diff = numpy.hstack((numpy.cumsum(model_coo_and_diff[:,0:2], axis=0), model_coo_and_diff))
        measurment_error = coordinates_and_diffs_uncut - model_coo_and_diff  # emg.dot(self.W.T)
        # To be used in the Kalman Filter as Ez, measurement noise
        self.measurment_error_covar = numpy.cov(measurment_error.T)
        
        self.A = numpy.linalg.pinv(coordinates_and_diffs_lag).dot(coordinates_and_diffs_now).T
        state_trans_error = coordinates_and_diffs_now - coordinates_and_diffs_lag.dot(self.A.T)
        # To be used in the Kalman Filter as Ex, process noise
        self.state_trans_covar = numpy.cov(state_trans_error.T)
    
    def predict(self, emg, cut_start=0, cut_end=0):
        transformed_emg = transform_emg(emg)
        transformed_emg = numpy.hstack((numpy.ones((transformed_emg.shape[0], 1)), transformed_emg))
        ticks_count = transformed_emg.shape[0]
        # TODO(kazeevn) how magic is 6?
        # We add zero preictions for the points we can't predict due to the lag
        # An exercise for the reader a better estimate is
        kalman_estimate = numpy.zeros((ticks_count + 2*EMG_FILTER_LAG, 6))
        wiener_estimate = numpy.zeros((ticks_count + 2*EMG_FILTER_LAG, 6))
        kalman_tick = numpy.zeros((6, 1))
        x_measurment_estimate_prev = numpy.zeros((1, 6))
        p_after = self.state_trans_covar.copy()
        for tick in range(1, transformed_emg.shape[0]):
            # Predict coordinate by state transition equation
            # TODO(kazeevn) why the name?
            x_state_estimate = self.A.dot(kalman_tick)
            # Predict MSE covarimance matrix estimate
            p_before = self.A.dot(p_after.dot(self.A.T)) + self.state_trans_covar
            # Predict coordinate by state measurement equation
            x_measurment_estimate = self.W.dot(transformed_emg[tick].T)[numpy.newaxis, :]
            # x_measurment_estimate = self.clf.predict(transformed_emg[tick].reshape(1,-1))
            # x_measurment_estimate[:,0][x_measurment_estimate[:,0]>3] = 3
            # x_measurment_estimate[:,1][x_measurment_estimate[:,1]>4] = 4
            x_measurment_estimate[x_measurment_estimate>4] = 4
            x_measurment_estimate[x_measurment_estimate<-4] = -4
            # x_measurment_estimate[:, 0:2] > 4 = numpy.min((x_measurment_estimate[:,0:2], [4]))
            # x_measurment_estimate[:, 0:2] = numpy.max((x_measurment_estimate[:,0:2], [-4]))
            x_measurment_estimate = numpy.hstack((x_measurment_estimate_prev[:,0:2] + x_measurment_estimate[:,0:2], x_measurment_estimate))
            p_after = numpy.linalg.pinv(numpy.linalg.pinv(p_before) +
                                        numpy.linalg.pinv(self.measurment_error_covar))
            # Update the state estimate
            kalman_tick = p_after.dot(numpy.linalg.pinv(p_before).dot(x_state_estimate) + 
                                          numpy.linalg.pinv(self.measurment_error_covar).dot(
                                            x_measurment_estimate.T))
            x_measurment_estimate_prev = x_measurment_estimate
            
            # Store the predicted coordinates
            kalman_estimate[tick + 2*EMG_FILTER_LAG] = kalman_tick.T
            wiener_estimate[tick + 2*EMG_FILTER_LAG] = x_measurment_estimate
        # kalman_coo = numpy.cumsum(kalman_estimate[:,0:2], axis=0)
        # kalman_estimate = numpy.hstack((kalman_coo, kalman_estimate))
        # wiener_coo = numpy.cumsum(wiener_estimate[:,0:2], axis=0)
        # wiener_estimate = numpy.hstack((wiener_coo, kalman_estimate))
        kalman_estimate[:,0:2] = savgol_filter(kalman_estimate[:,0:2].T, 7, 3).T
        mean_speed = numpy.mean(kalman_estimate[:,2:4], axis=0)
        wiener_estimate[:cut_start] = wiener_estimate[cut_start]
        kalman_estimate[:cut_start] = kalman_estimate[cut_start]
        if cut_end<-1:
            wiener_estimate[cut_end+1:] = wiener_estimate[cut_end]
            kalman_estimate[cut_end+1:] = kalman_estimate[cut_end]
        return (wiener_estimate, kalman_estimate)


# ## Validation

# In[9]:

def score_trial(true_coordinates, predicted_coordinates, dimensions=2):
    r =  numpy.mean([pearsonr(numpy.diff(true_coordinates[:, axis]),
                                numpy.diff(predicted_coordinates[:, axis]))[0] for
                       axis in range(dimensions)])
    return r


# In[10]:

numbergetter = re.compile(r"\d+$")
def get_tail_number(string):
    return int(numbergetter.findall(string)[0])


# In[11]:

shifts = [0, 1]
subject_ids = []
trials = []
wiener = []
kalman = []
# emg_add_bias = 1
out_list = pd.read_csv('out_list_7.csv')
out_set = set((s, d, t) for s, d, t in zip(out_list.subject, out_list.digit, out_list.trail))
with h5py.File(TRAIN_FILE, "r") as train_io, h5py.File(TEST_FILE, "r") as test_io:
    xy = list()
    i_list = list()
    j_list = list()
    for subject_id, subject_data in train_io.items():
        coordinates_and_diffs_list_all = []
        coordinates_list_all = []
        emg_list = []
        for digit, subject_trials in subject_data.items():
            if digit in('3','7','9','5'): # or (digit == '5' and subject_id[-1] != '1'):
                for trial_idx, trial in subject_trials.items():
                    if (subject_id, digit, trial_idx) not in out_set:
                        pen_coordintes_with_emg_lag = trial['pen_coordinates']
                        ar = numpy.array(pen_coordintes_with_emg_lag)
                        ar, i, j = cut_steady(ar, 2)
                        if ar.shape[0]<15:
                            continue
                        if digit != '1':
                            i_list.append(i)
                            j_list.append(j)
                        emg = numpy.array(trial['emg'])
                        emg = emg[i:j,:]
                        if digit == '5':
                            ar, j = interpolate_steady(ar, 2)
                            emg = emg[:j,:]
                        coordinates_list_all.append(ar)
                        # ar = ar[:-emg_add_bias,:]
                        # emg = emg[emg_add_bias:,:]
                        ar = numpy.hstack(align_arrays(
                                compute_differentials(ar, 2, 0)))
                        coordinates_and_diffs_list_all.append(ar)
                        emg_list.append(emg)
                        assert(trial['emg'].shape[0] == trial['pen_coordinates'].shape[0])
        # Validataion
        model = BaselineModel()
        emg_train, emg_test, coordinates_and_diffs_list_train, coordinates_and_diffs_list_test, coordinates_list_train, coordinates_list_test =             train_test_split(emg_list, coordinates_and_diffs_list_all, coordinates_list_all)
        model.fit(emg_train, coordinates_and_diffs_list_train)
        score_wiener = 0
        score_kalman = 0
        for trial_emg, trial_coordinates_and_diffs in zip(emg_test, coordinates_list_test):
            wiener_estimate, kalman_estimate = model.predict(trial_emg, 0, 0)
            # 2: is due to aligning of the arrays while computing the differentials.
            # For a perfectly coorrect validation (like in the contest) it of course should
            # be done over the entire arrays
            score_wiener += score_trial(trial_coordinates_and_diffs, wiener_estimate)
            score_kalman += score_trial(trial_coordinates_and_diffs, kalman_estimate)
        validation_count = len(emg_test)
        print("Mean scores; Kalman: %f, Wiener: %f" % (score_kalman/validation_count, score_wiener/validation_count))
        # Prediction
        model = BaselineModel()
        model.fit(emg_list, coordinates_and_diffs_list_all)
        for trial_name, trial_data in test_io[subject_id].items():
            trials.append(get_tail_number(trial_name))
            subject_ids.append(get_tail_number(subject_id))
            cut_start = round(sum(i_list)/len(i_list)) if len(i_list) else 0
            cut_end = round(sum(j_list)/len(j_list)) if len(j_list) else 0
            wiener_estimate, kalman_estimate = model.predict(trial_data, cut_start, cut_end)
            kalman.append(kalman_estimate)
            wiener.append(wiener_estimate)
            xy.append(kalman_estimate[:,0:2])
    matdict = dict()
    matdict['xy'] = xy
    sio.savemat('C:/test.mat', matdict)


# In[12]:

def write_solution(trial_indexes, trajectories, file_name):
    solution_list = numpy.vstack([
            [[subject_id, trial_index, tick, xy[0], xy[1]] for tick, xy in enumerate(trajectory)] for
            subject_id, trial_index, trajectory in zip(subject_ids, trial_indexes, trajectories)
        ])
    solution = pd.DataFrame.from_records(solution_list, columns=["subject_id", "trial_id", "tick_index", "x", "y"])
    solution.to_csv(file_name, index=False)


# In[13]:

write_solution(trials, kalman, BASELINE_KALMAN)
# write_solution(trials, wiener, BASELINE_WIENER)


# In[ ]:



