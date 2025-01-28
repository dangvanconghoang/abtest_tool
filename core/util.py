import ast
import json
import logging
import math
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAX_CONVERSIONS = 800000


class JsonSerializable(object):
    """ Interface for serializable classes."""
    print('dont print_variable')
    # def dumper(self):
    #     try:
    #         return self.toJSON()
    #     except:
    #         return string(self)

    def toJson(self):
        return json.dumps(self, default=lambda o: o.name if isinstance(o, Enum) else o.__dict__, sort_keys=True, indent=4)
        # return json.dumps(self,sort_keys=True,default=self.dumper(), indent=4)        

    def __repr__(self):
        return self.toJson()


def find_value_by_key_with_condition(items, condition_key, condition_value, lookup_key):
    """ Find the value of lookup key where the dictionary contains condition key = condition value.
    
    :param items: list of dictionaries
    :type  items: list
    :param condition_key: condition key
    :type  condition_key: str
    :param condition_value: a value for the condition key
    :param lookup_key: lookup key or key you want to find the value for
    :type  lookup_key: str
    
    :return: lookup value or found value for the lookup key
    """
    return [item[lookup_key] for item in items if item[condition_key] == condition_value][0]


def is_nan(obj):
    """ Checks whether the input is NaN. It uses the trick that NaN is not equal to NaN."""
    return obj != obj


def drop_nan(array):
    """ Drop Nan values from the given numpy array.
    
    :param array: input array
    :type  array: np.ndarray
    
    :return: a new array without NaN values
    :rtype: np.ndarray
    """
    if array.ndim == 1:
        return array[~np.isnan(array)]
    elif array.ndim == 2:
        return array[~np.isnan(array).any(axis=1)]


def generate_random_data(seed=42):
    """ Generate random data for two variants. It can be used in unit tests or demo. 
    The function generates a dataset with the following columns:
    - entity: unique identifier for each data point
    - variant: A or B
    - normal_same: random normal data for both variants
    - normal_shifted: random normal data with different mean for each variant
    - feature: a feature column with two values: has and non
    - normal_shifted_by_feature: random normal data with different mean for each variant and feature
    - treatment_start_time: an integer column with values between 0 and 9
    - normal_unequal_variance: random normal data with different standard deviation for each variant
    
    :param seed: random seed for reproducibility
    :type  seed: int
    
    :return: a dataframe with random data and metadata
    :rtype: pd.DataFrame
    """
    np.random.seed(seed)
    size = 10000

    data = pd.DataFrame()
    data['entity'] = list(range(size))
    data['variant'] = np.random.choice(['A', 'B'], size=size, p=[0.6, 0.4])

    data['normal_same'] = np.random.normal(size=size)
    data['normal_shifted'] = np.random.normal(size=size)

    size_shifted_B = data['normal_shifted'][data['variant'] == 'B'].shape[0]
    data.loc[data['variant'] == 'B', 'normal_shifted'] = np.random.normal(loc=1.0, size=size_shifted_B)

    data['feature'] = np.random.choice(['has', 'non'], size=size)
    data.loc[0, 'feature'] = 'feature that only has one data point'
    data['normal_shifted_by_feature'] = np.random.normal(size=size)

    ii = (data['variant'] == 'B') & (data['feature'] == 'has')
    randdata_shifted_mean = np.random.normal(loc=1.0, size=sum(ii == True))
    data.loc[ii, 'normal_shifted_by_feature'] = randdata_shifted_mean

    data['treatment_start_time'] = np.random.choice(list(range(10)), size=size)
    data['normal_unequal_variance'] = np.random.normal(size=size)

    size_unequalvar_B = data['normal_unequal_variance'][data['variant'] == 'B'].shape[0]
    data.loc[data['variant'] == 'B', 'normal_unequal_variance'] = np.random.normal(scale=10, size=size_unequalvar_B)

    # Add date column
    d1 = datetime.strptime('2022-01-01', '%Y-%m-%d')
    d2 = datetime.strptime('2023-03-01', '%Y-%m-%d')
    date_col = []
    delta = d2 - d1
    for i in range(delta.days * 24 + 1):
        date_col.append((d1 + timedelta(hours=i)).strftime('%Y-%m-%d'))
    data['date'] = date_col[:size]

    metadata = {
        'primary_KPI': 'normal_shifted',
        'source': 'simulated',
        'experiment': 'random_data_generation'
    }
    return data, metadata


def lbeta(x, y):
    """
    Calculate the log beta function.
    
    :param x: value for x
    :type  x: float
    :param y: value for y
    :type  y: float
    
    :return: the value of the log beta function
    :rtype: float
    """
    return math.lgamma(x) +math.lgamma(y) - math.lgamma(x+y)


def first_crossing_one_sided_pdf(n, z, log_p, log_1_p):
    """
    Calculate the first crossing one-sided pdf.
    
    :param n: sample size
    :type  n: float
    :param z: z-score
    :type  z: float
    :param log_p: log of p
    :type  log_p: float
    :param log_1_p: log of 1-p
    :type  log_1_p: float
    
    :return: the first crossing one-sided pdf
    :rtype: float
    """
    k = 0.5*(n+z)
    return z /n / k * math.exp(-lbeta(k, n+1-k) + (k-z)*log_p + k*log_1_p)

def bsearch(z_lo, z_hi, alpha, power_level, null_p, alt_p):
    """
    Binary search algorithm to find the z-score for a given alpha and power level. The function returns the z-score and the upper bound of the z-score.
    
    :param z_lo: lower bound of the z-score
    :type  z_lo: float
    :param z_hi: upper bound of the z-score
    :type  z_hi: float
    :param alpha: significance level
    :type  alpha: float
    :param power_level: power level
    :type  power_level: float
    :param null_p: null hypothesis conversion rate
    :type  null_p: float
    :param alt_p: alternative hypothesis conversion rate
    :type  alt_p: float
    
    :return: z-score and upper bound of the z-score
    :rtype: float, float
    """
    log_null_p = math.log(null_p)
    log_null_1_p = math.log(1.0-null_p)

    log_alt_p = math.log(alt_p)
    log_alt_1_p = math.log(1.0-alt_p)

    z = z_lo+2*math.floor((z_hi-z_lo)/4)
    while (z_lo < z_hi):
        null_cdf = 0.0
        alt_cdf = 0.0
        old_lo = z_lo
        old_hi = z_hi
        n=z
        for n in range(z,MAX_CONVERSIONS+1,2):
            k = 0.5*(n+z)
            prefix = z / n / k
            lbeta_k = lbeta(k,n+1-k)
            null_cdf += prefix * math.exp(-lbeta_k + (k-z)*log_null_p + k*log_null_1_p)
            alt_cdf += prefix * math.exp(-lbeta_k + (k-z)*log_alt_p + k*log_alt_1_p)
            if (math.isnan(null_cdf) | math.isnan(alt_cdf)):
                break
            
            if (alt_cdf > power_level):
                if (null_cdf < alpha):
                    z_hi = z
                else :
                    z_lo = z+2
                
                break
            elif  (null_cdf > alpha):
                z_lo = z+2
                break
            
        
        if (math.isnan(null_cdf) | math.isnan(alt_cdf) | n >= MAX_CONVERSIONS):
            # print("NaN...")
            break
        
        z = z_lo + 2*math.floor((z_hi-z_lo)/4)
    
    return z,z_hi

def n_for_z(z, alpha, power_level, null_p, alt_p):
    """
    Calculate the sample size for a given z-score. The function uses the binary search algorithm to find the sample size.
    
    :param z: z-score
    :type  z: float
    :param alpha: significance level
    :type  alpha: float
    :param power_level: power level
    :type  power_level: float
    :param null_p: null hypothesis conversion rate
    :type  null_p: float
    :param alt_p: alternative hypothesis conversion rate
    :type  alt_p: float
    
    :return: sample size
    :rtype: float
    """
    n = np.nan
    null_cdf = 0.0
    alt_cdf = 0.0

    log_null_p = math.log(null_p)
    log_null_1_p = math.log(1.0-null_p)

    log_alt_p = math.log(alt_p)
    log_alt_1_p = math.log(1.0-alt_p)

    n=z

    for n in range(z,MAX_CONVERSIONS+1,2):
        k = 0.5*(n+z)
        prefix = z / n / k
        lbeta_k = lbeta(k,n+1-k)
        null_cdf += prefix * math.exp(-lbeta_k + (k-z)*log_null_p + k*log_null_1_p)
        alt_cdf += prefix * math.exp(-lbeta_k + (k-z)*log_alt_p + k*log_alt_1_p)
        if (math.isnan(null_cdf) | math.isnan(alt_cdf)):
            return np.nan
        
        if (alt_cdf > power_level):
            if (null_cdf < alpha):
                return n
            else:
                return np.nan
  
    return np.nan



def clean_json(json=''):
    """
    Clean the json string. It replaces some values to be able to parse it.
    
    :param json: json string
    :type  json: str
    
    :return: cleaned json
    :rtype: dict
    """
    if json in ['', 'None', 'null', 'NULL', 'nan', 'NaN']:
        return ''
    
    json = json.replace("'", '"')
    json = json.replace("-inf", '"-inf"')
    json = json.replace(" inf", '"inf"')
    json = json.replace("False", '"False"')
    json = json.replace("True", '"True"')
    json = json.replace("nan", 'None')
    json = ast.literal_eval((json))
    return json



def check_balance_sample(exp,result_test, data = pd.DataFrame(), weight_traffic = {}):
    """
    Check the balance of the sample size for each variant. The function calculates the chi-square test with the expected traffic of each variant. 
    
    :param exp: experiment instance
    :type  exp: Experiment
    :param result_test: result of the test
    :type  result_test: dict
    :param data: data for the test
    :type  data: pd.DataFrame
    :param weight_traffic: expected traffic for each variant
    :type  weight_traffic: dict
    
    :return: the result of the test with the traffic balance
    :rtype: dict
    """
    if data.empty:
        data = exp.data
    if weight_traffic == {}:
        weight_traffic = exp.weight_traffic
      # weight_traffic=  {
      # 'A' : 0.33 ,
      # 'B' : 0.33 ,
      # 'C' : 0.33 ,
      # }
    
    list_variants = data['variant'].unique().tolist()

    list_size_cfg = []
    list_size_actual = []
    for i in  list_variants:
        size_i_cfg = weight_traffic[i]
        list_size_cfg.append(size_i_cfg)

        size_i_actual = data[data['variant'] == i].shape[0]
        list_size_actual.append(size_i_actual)


    observed_freqs = pd.Series(list_size_actual, list_variants)
    expected_freqs = pd.Series(list_size_cfg  , list_variants)* np.sum(observed_freqs)
    is_balance, p_value = exp.run_goodness_of_fit_test( 
        observed_freqs=observed_freqs, expected_freqs=expected_freqs)
    result_test['traffic'] = {'is_balance': is_balance,
                                'p_value': p_value}
    # logging.info('check traffic imbalance: ' +
    #                 str(is_balance) + '; p_value: ' + str(p_value))
    return result_test


def calculate_precision(result_test,expected_precision):
    """
    Calculate the precision of the test and the progress of the precision compared to the expected precision.
    
    :param result_test: result of the test
    :type  result_test: dict
    :param expected_precision: expected precision
    :type  expected_precision: float
    
    :return: the result of the test with the precision and its progress
    :rtype: dict
    """
    delta_pct = result_test['results'][0]['result']['original_test_statistics']['delta']/result_test['results'][0]['result']['original_test_statistics']['control_statistics']['mean']
    ci_l_pct = float(result_test['results'][0]['result']['original_test_statistics']['confidence_interval'][0]['value'])/result_test['results'][0]['result']['original_test_statistics']['control_statistics']['mean']
    # ci_u_pct = float(exp_result['result']['confidence_interval'][1]['value'])/exp_result['result']['control_mean']
    precision = delta_pct - ci_l_pct
    precision_progress = math.pow((expected_precision/precision), 2)
    result_test['precision'] = {'precision': precision,
                        'precision_progress': precision_progress}
    return result_test

    

def result_test_to_df(exp_id, metric_id, primary_metric_result, list_secondary_metric_result_before_correction, list_secondary_metric_result_after_correction):
    """
    Convert the result of the test to a dataframe for saving to the database.
    
    :param exp_id: experiment id
    :type  exp_id: str
    :param metric_id: metric id
    :type  metric_id: str
    :param primary_metric_result: primary metric result
    :type  primary_metric_result: float
    :param list_secondary_metric_result_before_correction: list of secondary metric results before correction
    :type  list_secondary_metric_result_before_correction: list
    :param list_secondary_metric_result_after_correction: list of secondary metric results after correction
    :type  list_secondary_metric_result_after_correction: list
    
    :return: a dataframe for saving to the database
    :rtype: pd.DataFrame
    """
    df_result = pd.DataFrame({
        'update_timestamp': [datetime.now()],
        'exp_id': [exp_id],
        'metric_id': [metric_id],
        'primary_metric_result': [str(primary_metric_result)],
        'list_secondary_metric_result_before_correction': [str(list_secondary_metric_result_before_correction)],
        'list_secondary_metric_result_after_correction': [str(list_secondary_metric_result_after_correction)],

    })
    return df_result
