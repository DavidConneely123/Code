import numpy as np
import matplotlib.pyplot as plt



# system_dividing = 2 ==> 10:(5,5) , 12:(6,6) etc...
# iterations = 100

x1 = [2,3,4,5,6,7,8,9,10,12,14,16,18]
y1 =[35.44, 24.15, 19.61, 17.16, 17.63, 14.51, 13.1, 12.08, 11.66, 10.32, 9.64, 8.51, 8.27]


# system-diving = 4 ==> 4:(1,3) , 6:(2,4) , 8:(2,6) , 10:(3,7) ... (always ceil(number_of_nuclei/system_dividing) nuclei in first group
# note: still 2:(1,1) so this should be the same

x2 = [2,4,6,8,10,12,14,16]
y2 = [35.57, 21.07, 14.79, 12.03, 11.38, 10.07, 8.58, 8.00]


mymodel = np.poly1d(np.polyfit(x1, y1, 1))
myline = np.linspace(0, 20, 1000)


plt.plot(x1,y1, label = 'System Dividing = 2')
plt.plot(x2,y2, label = 'System Dividing = 4')
plt.xlabel('Number of Nuclei')
plt.ylabel('Mean Overestimation of Vmax (%)')
plt.title('Variation in Vmax Overestimation with increasing Number of Nuclei \n (100 Iterations Each)')
plt.ylim((0,50))
plt.xlim(2,30)
plt.legend()


#plt.plot(myline, mymodel(myline))


plt.show()