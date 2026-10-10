import unittest
import numpy as np
import pandas as pd
from scripts.audit_inputs import numeric_clean, correlation_stats, aviation_fields

class AnalysisTests(unittest.TestCase):
    def test_aviation_names_exclude_country_and_ratios(self):
        columns=['Departure Country Name','Total PAX','Transit First Count','Economy Class Seat Capacity','Load Factor','Average Weekly Frequency']
        self.assertEqual(aviation_fields(columns),['Total PAX','Transit First Count','Economy Class Seat Capacity'])

    def test_suppression_is_not_zero(self):
        actual=numeric_clean(pd.Series(['*', '', '  *  ', None, '0', '12.5', 'invalid']))
        self.assertTrue(actual.iloc[:4].isna().all())
        self.assertEqual(actual.iloc[4],0)
        self.assertEqual(actual.iloc[5],12.5)
        self.assertTrue(pd.isna(actual.iloc[6]))

    def test_pairwise_missing_count(self):
        result=correlation_stats(pd.Series([1,2,3,np.nan,5]),pd.Series([2,4,6,8,np.nan]))
        self.assertEqual(result[2],3)
        self.assertAlmostEqual(result[0].statistic,1)
        self.assertAlmostEqual(result[1].statistic,1)

    def test_constant_or_insufficient_not_reported(self):
        self.assertIsNone(correlation_stats(pd.Series([1,1,1]),pd.Series([2,3,4])))
        self.assertIsNone(correlation_stats(pd.Series([1,2]),pd.Series([2,3])))

    def test_rank_and_linear_association_differ(self):
        pearson,spearman,n=correlation_stats(pd.Series([1,2,3,4,5]),pd.Series([1,4,9,16,25]))
        self.assertEqual(n,5)
        self.assertAlmostEqual(spearman.statistic,1)
        self.assertLess(pearson.statistic,.99)

if __name__=='__main__': unittest.main()
