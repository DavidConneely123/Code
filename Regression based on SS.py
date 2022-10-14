import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import scipy
from scipy.stats import linregress
from sklearn import datasets, linear_model



data1 = pd.read_csv('3 Nuclei - 3276 Iterations - SS')
data2 = pd.read_csv('4 Nuclei - 20475 Iterations - SS')
data3 = pd.read_csv('5 Nuclei - 10000 Iterations - SS')
data4 = pd.read_csv('6 Nuclei - 600 Iterations - SS')
data5 = pd.read_csv('7 Nuclei - 600 Iterations - SS')
data6 = pd.read_csv('8 Nuclei - 600 Iterations - SS')
data7 = pd.read_csv('9 Nuclei - 600 Iterations - SS')
data8 = pd.read_csv('10 Nuclei - 1000 Iterations - SS')
data9 = pd.read_csv('11 Nuclei - 600 Iterations - SS')
data10 = pd.read_csv('12 Nuclei - 600 Iterations - SS')
data11 = pd.read_csv('13 Nuclei - 600 Iterations - SS')
data12 = pd.read_csv('15 Nuclei - 600 Iterations - SS')
data13 = pd.read_csv('16 Nuclei - 600 Iterations - SS')

combined_data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13])

mymodel = np.poly1d(np.polyfit(combined_data['x'], combined_data['y'], 2))
myline = np.linspace(0, 1.355e16, 1000)

colors = cm.rainbow(np.linspace(0,1,20))

print(mymodel.coef)

exactdata = [i for i in pd.read_csv('Building up Vmax with Nuclei - Decreasing SS - 21 Nuclei')['Vmax']]  # Importing the exactly calculated Vmax values
SS_list = [i for i in pd.read_csv('Sum of Squares Dictionary')['0']]
SS_sumlist = [np.sum(SS_list[0:i]) for i in range(28)] # A list of the sum of the squares of the HFI up to a given nucleus





plt.scatter(data1['x'], data1['y'], color = cm.rainbow(1/15))
plt.scatter(data2['x'], data2['y'], color = cm.rainbow(2/15))
plt.scatter(data3['x'], data3['y'], color = cm.rainbow(3/15))
plt.scatter(data4['x'], data4['y'], color = cm.rainbow(4/15))
plt.scatter(data5['x'], data5['y'], color = cm.rainbow(5/15))
plt.scatter(data6['x'], data6['y'], color = cm.rainbow(6/15))
plt.scatter(data7['x'], data7['y'], color = cm.rainbow(7/15))
plt.scatter(data8['x'], data8['y'], color = cm.rainbow(8/15))
plt.scatter(data9['x'], data9['y'], color = cm.rainbow(9/15))
plt.scatter(data10['x'], data10['y'], color = cm.rainbow(10/15))
plt.scatter(data11['x'], data11['y'], color = cm.rainbow(11/15))
plt.scatter(data12['x'], data12['y'], color = cm.rainbow(12/15))
plt.scatter(data13['x'], data13['y'], color = cm.rainbow(13/15))

plt.scatter(SS_sumlist[0:22], exactdata, color = 'black')

plt.plot(myline, mymodel(myline), label = 'Regression Based on Calculations with Smaller Systems', color = 'purple', linewidth = 2)
plt.plot(SS_sumlist[0:22], exactdata, label = 'Exact Calculations', color = 'black')
plt.plot(np.linspace(1.3520317431856022e+16, 1.3520317431856022e+16, 100), np.linspace(0,300,100), '-.', color = 'green')



plt.ylabel('Vmax (MHz)')
plt.xlabel('Sum of Squares of HFI Tensor Elements (Amn^2)')

plt.legend()
plt.show()

#slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(combined_data['x'], combined_data['y'])
#print(r_value)

#coefficents_of_fitting = np.polyfit(combined_data['x'], combined_data['y'], 1)
#fitting = np.poly1d(coefficents_of_fitting)
#print(fitting)