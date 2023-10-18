import math
import scipy
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

ALPHA = 0.01

# question 1
print("Question 1:")
data = pd.DataFrame(pd.read_csv("./Crime_R.csv"))

# part a
crime_rates = data[['CrimeRate', 'Southern']].to_numpy()
southern_crime_rates = []
not_southern_crime_rates = []

for i in range(len(crime_rates)):
    if crime_rates[i][1] == 1:
        southern_crime_rates.append(crime_rates[i][0])
    else:
        not_southern_crime_rates.append(crime_rates[i][0])

crime_rates1 = southern_crime_rates
crime_rates2 = not_southern_crime_rates
sample_size1, sample_size2 = len(crime_rates1), len(crime_rates2)
mean1, mean2 = np.mean(crime_rates1), np.mean(crime_rates2)
standard_deviation1, standard_deviation2 = np.std(crime_rates1), np.std(crime_rates2)
denominator = math.sqrt(standard_deviation1 ** 2 / sample_size1 + standard_deviation2 ** 2 / sample_size2)
z_value_a = (mean1 - mean2) / denominator
p_value_a = 1 - scipy.stats.norm(0,1).cdf(z_value_a)

print("\tA)", p_value_a, (p_value_a <= ALPHA))

# part b
crimerate = data[['CrimeRate']].to_numpy()
crimerate10 = data[['CrimeRate10']].to_numpy()
sample_size1, sample_size2 = len(crimerate), len(crimerate10)
mean1, mean2 = np.mean(crimerate), np.mean(crimerate10)
standard_deviation1, standard_deviation2 = np.std(crimerate), np.std(crimerate10)
denominator = math.sqrt(standard_deviation1 ** 2 / sample_size1 + standard_deviation2 ** 2 / sample_size2)
z_value_b = (mean1 - mean2) / denominator
p_value_b = 1 - scipy.stats.norm(0,1).cdf(z_value_b)

print("\tB)", p_value_b, (p_value_b <= ALPHA))

# part c
education = data[['CrimeRate', 'Education']].to_numpy()
education_set1 = []
education_set2 = []
education_set3 = []

for i in range(len(education)):
    if education[i][1] > 13:
        education_set3.append(education[i][0])
    elif education[i][1] < 11:
        education_set1.append(education[i][0])
    else:
        education_set2.append(education[i][0])

anova_test = scipy.stats.f_oneway(education_set1, education_set2, education_set3)

print("\tC)", anova_test[1], (anova_test[1] <= ALPHA))

# part d
relationship = data[['HighYouthUnemploy', 'Southern']].to_numpy()
relationship1=[]
relationship2=[]
frequency10 = 0
frequency11 = 0
frequency20 = 0
frequency21 = 0

for i in range(len(relationship)):
    relationship1.append(relationship[i][0])
    relationship2.append(relationship[i][1])
    if relationship[i][0] == 0:
        frequency10  += 1
    elif relationship[i][0] == 1:
        frequency11 += 1
    elif relationship[i][1] == 0:
        frequency20  += 1
    elif relationship[i][1] == 1:
        frequency21  += 1

frequencies = [frequency10 , frequency11, frequency20 , frequency21 ]
chi_squared_test = scipy.stats.chisquare(frequencies)
print("\tD)", chi_squared_test[1])
print()

# question 2
print("Question 2:")

data1 = pd.DataFrame(pd.read_csv("./07-07-2021.csv"))
data2 = pd.DataFrame(pd.read_csv("./07-07-2022.csv"))
case_fatality1 = data1[['Case_Fatality_Ratio']].to_numpy()
case_fatality2 = data2[['Case_Fatality_Ratio']].to_numpy()
case_fatality1.flatten()
case_fatality2.flatten()
copy1 = []
copy2 = []

for i in range(len(case_fatality1)):
    if not math.isnan(case_fatality1[i]):
        copy1.append(case_fatality1[i])

for i in range(len(case_fatality2)):
    if not math.isnan(case_fatality2[i]):
        copy2.append(case_fatality2[i])

case_fatality1 = copy1
case_fatality2 = copy2
sample_size1, sample_size2 = len(case_fatality1), len(case_fatality2)
mean1, mean2 = np.mean(case_fatality1), np.mean(case_fatality2)
standard_deviation1, standard_deviation2 = np.std(case_fatality1), np.std(case_fatality2)
denominator = math.sqrt(standard_deviation1 ** 2 / sample_size1 + standard_deviation2 ** 2 / sample_size2)
z_test = (mean1 - mean2) / denominator
z_test_alpha = scipy.stats.norm.ppf(ALPHA)

print("\tA)", z_test, z_test_alpha, (z_test >= z_test_alpha))
