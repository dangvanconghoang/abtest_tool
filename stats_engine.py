import json
import logging
from pprint import pprint
from statsmodels.stats.multitest import multipletests

import pandas as pd
# from ab_test.utils.stats import *

from utils.abtest_core.core.experiment import Experiment as Expan_Experiment
from utils.abtest_core.core.statistical_test import (KPI, CorrectionMethod,
                                                     StatisticalTest,
                                                     StatisticalTestSuite,
                                                     Variants)
from utils.abtest_core.core.statistics import estimate_sample_size
from utils.abtest_core.core.util import *

class Experiment:
    """
    Experiment class to run AB Test
    """
    def __init__(self, data, exp_cfg):

        self.data = pd.DataFrame(data)
        self.exp_cfg = exp_cfg

        self.primary_metric_result = {}

        self.list_secondary_metric_result_before_correction = []
        self.list_secondary_metric_result_after_correction = []
        self.df_final_result = []

        self.exp_name = self.exp_cfg['exp_name']
        self.list_metrics = self.exp_cfg['list_metrics']

        self.primary_metric = self.exp_cfg['primary_metric']
        self.list_secondary_metrics = [x1 for x1 in self.list_metrics if x1 !=  self.primary_metric]
        self.list_original_pvalue_secondary_metrics = []

        self.weight_traffic = self.exp_cfg['weight_traffic']
        self.test_method = self.exp_cfg['test_method']
        self.control_variant = self.exp_cfg['control_variant']
        self.mde = self.exp_cfg['mde']
        self.alpha = self.exp_cfg['alpha']
        self.list_variants = self.data['variant'].unique().tolist()
        self.beta = self.exp_cfg['beta']
        self.expected_precision = self.exp_cfg['expected_precision']
        self.precision = None
        self.precision_progress = None


    def is_binary_column(self, df, column_name):
        """
        Check if the column is binary or not
        :param df: dataframe
        :param column_name: column name
        :return: 'binary' if binary, 'normal' if not
        """
        if df[column_name].nunique() == 2:
            return 'binary'
        return 'normal'

    
    def execute_abtest(self,metric_name = '',  alpha = 0.05, correct = True):
        """
        Execute AB Test
        :param metric_name: metric name
        :param alpha: alpha
        :param correct: correction or not
        :return: result of AB Test
        """
        logging.info("run test with "+str(len(self.data))+" rows")    

        #lib settup requirement
        dimensions = []
        metadata = {
            'primary_KPI':  metric_name,
            'source': '---',
            'experiment': self.exp_name
        }

        kpi = KPI(metric_name)
        exp_expan = Expan_Experiment(metadata=metadata)


        #generate a list of combinations len(variants)C2 
        from itertools import combinations         
        # Generate all combinations of two elements
        all_combinations = list(combinations(self.list_variants, 2))
        # Filter combinations to include only those that contain 'control_variant'
        variants = [list(comb) for comb in all_combinations if self.control_variant in comb]
        
        tests= []
        for v in variants:
            # test = StatisticalTest(
            #     data=data, kpi=kpi, features=dimensions, variants=variants)
            test = StatisticalTest(data[data["variant"].isin(v)],   kpi, [], Variants(variant_column_name='variant', control_name=v[0], treatment_name=v[1]))
            tests.append(test)

        if correct == False:
            test_suite = StatisticalTestSuite(tests=tests, correction_method=CorrectionMethod.NONE)
        else:
            test_suite = StatisticalTestSuite(tests=tests, correction_method=CorrectionMethod.BH)
 
        endpoint = self.is_binary_column(data, metric_name)

        ess = estimate_sample_size(x=data[data['variant'] == self.control_variant][metric_name],
                                   mde=self.mde, r= 1 , alpha= alpha, beta=self.beta, endpoint=endpoint)
        logging.info('estimate_sample_size: '+str(ess))
        
        if ess < 500:
            ess = 500

        # if self.test_method_cfg == 'Sequencial':
        if endpoint == 'binary':
            expgroup_sequential = exp_expan.analyze_statistical_test_suite(
                test_suite=test_suite, test_method='group_sequential', estimated_sample_size=ess,  alpha= alpha,  endpoint='binary')
    
        if endpoint == 'normal':
            expgroup_sequential = exp_expan.analyze_statistical_test_suite(
                test_suite=test_suite, test_method='group_sequential', estimated_sample_size=ess, alpha= alpha,   endpoint='normal')
        
        
        expgroup_sequential = str(expgroup_sequential).replace('\n', '')
        result_test = json.loads(expgroup_sequential)
        
        # check is balance?
        result_test = check_balance_sample(
            exp_expan,result_test, data = data, weight_traffic = self.weight_traffic)

        #calculate precision
        result_test = calculate_precision(result_test = result_test,expected_precision = self.expected_precision)

        pprint(result_test)
        
        return result_test
    

    # generate multiple test from multiple metrics
    def generate_multiple_tests(self):
        """
        Generate multiple tests
        """
        # # primary metric
        logging.info('primary metric is calculated '+self.primary_metric)
        self.primary_metric_result = self.execute_abtest(self.primary_metric, alpha = self.alpha)

        # secondary metrics
        logging.info('secondary metrics is calculated ' )
        print((self.list_secondary_metrics))
        for m in self.list_secondary_metrics:
            test_result_m = self.execute_abtest(m, self.alpha)
            self.list_secondary_metric_result_before_correction.append(test_result_m)
            # list_original_pvalue_secondary_metrics
            for i in range(len(test_result_m['results'])):
                self.list_original_pvalue_secondary_metrics.append(test_result_m['results'][i]['result']['original_test_statistics']['p'])

        logging.info('list_original_pvalue_secondary_metrics' + str(self.list_original_pvalue_secondary_metrics))
 
        reject, pvals_corrected, _, alphacBonf = multipletests(self.list_original_pvalue_secondary_metrics, alpha=self.alpha, method='fdr_bh')
        
        logging.info('alpha after correction: '+str(alphacBonf))

        # rerun with new alpha
        logging.info('rerun with new alpha secondary metrics' )
        self.list_secondary_metric_result_after_correction = []
        for m in self.list_secondary_metrics:
            self.list_secondary_metric_result_after_correction.append(self.execute_abtest(m, alpha = alphacBonf, correct = False))

    def main(self):
        """
        Main function to run AB Test
        """
        self.generate_multiple_tests()
        self.df_final_result = result_test_to_df(self.exp_name,self.list_metrics , self.primary_metric_result, self.list_secondary_metric_result_before_correction, self.list_secondary_metric_result_after_correction)