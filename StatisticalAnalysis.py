# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:04:46 2023

@author: adeyem01
"""

import numpy as np
from scipy.stats import ttest_rel, shapiro, levene

# Accuracy values for the two datasets
accuracy_full = np.array([65.53, 52.33, 76.31, 77.58, 77.23, 72.62])
accuracy_reduced = np.array([64.73, 68.41, 74.75, 74.02, 74.43, 71.62])

# Calculate differences in accuracy
differences = accuracy_full - accuracy_reduced

# Check assumptions
# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = shapiro(differences)
print("Shapiro-Wilk test for normality:")
print("Statistic:", shapiro_stat)
print("p-value:", shapiro_p)
if shapiro_p < 0.05:
    print("The differences are not normally distributed.")
else:
    print("The differences are normally distributed.")

# Levene's test for homogeneity of variances
levene_stat, levene_p = levene(accuracy_full, accuracy_reduced)
print("\nLevene's test for homogeneity of variances:")
print("Statistic:", levene_stat)
print("p-value:", levene_p)
if levene_p < 0.05:
    print("The variances are not equal.")
else:
    print("The variances are equal.")

# Perform paired t-test
t_stat, p_value = ttest_rel(accuracy_full, accuracy_reduced)
print("\nPaired t-test:")
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Interpret results
alpha = 0.05
if p_value < alpha:
    print("\nReject null hypothesis: There is a significant difference in accuracy.")
else:
    print("\nFail to reject null hypothesis: There is no significant difference in accuracy.")
