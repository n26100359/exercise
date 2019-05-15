# library
import csv
from numpy import genfromtxt
import numpy as np
my_data = genfromtxt('deoxys.csv', delimiter=',')
a = [1,5,9,12,18,20,22,23,24,25]

x_dict = {}
y_dict = {}
for i in range(1,10):
    x_dict[i] = np.zeros(shape=(20,3,a[i]-a[i-1]))
    y_dict[i] = np.zeros(shape=(20,1,a[i]-a[i-1]))
for i in range(1,10):
    for j in range(0,20):
        x_dict[i][j] = my_data[1+j:4+j,a[i-1]:a[i]]
        y_dict[i][j] = my_data[4+j,a[i-1]:a[i]]
print(x_dict[1][19])
#with open('x.pkl', 'wb') as f:
#    pkl.dump(x_all, f)
#train1 = np.zeros(shape=(20,3,a[1]-a[0]))
#test1 = np.zeros(shape=(20,1,a[1]-a[0]))
#train2 = np.zeros(shape=(20,3,a[2]-a[1]))
#test2 = np.zeros(shape=(20,1,a[2]-a[1]))
#train3 = np.zeros(shape=(20,3,a[3]-a[2]))
#test3 = np.zeros(shape=(20,1,a[3]-a[2]))
#train4 = np.zeros(shape=(20,3,a[4]-a[3]))
#test4 = np.zeros(shape=(20,1,a[4]-a[3]))
#train5 = np.zeros(shape=(20,3,a[5]-a[4]))
#test5 = np.zeros(shape=(20,1,a[5]-a[4]))
#train6 = np.zeros(shape=(20,3,a[6]-a[5]))
#test6 = np.zeros(shape=(20,1,a[6]-a[5]))
#train7 = np.zeros(shape=(20,3,a[7]-a[6]))
#test7 = np.zeros(shape=(20,1,a[7]-a[6]))
#train8 = np.zeros(shape=(20,3,a[8]-a[7]))
#test8 = np.zeros(shape=(20,1,a[8]-a[7]))
#train9 = np.zeros(shape=(20,3,a[9]-a[8]))
#test9 = np.zeros(shape=(20,1,a[9]-a[8]))
#for i in range(0,20):
#    train1[i] = my_data[1+i:4+i,a[0]:a[1]]
#    test1[i] = my_data[4+i,a[0]:a[1]]
#for i in range(0,20):
#    train2[i] = my_data[1+i:4+i,a[1]:a[2]]
#    test2[i] = my_data[4+i,a[1]:a[2]]
#for i in range(0,20):
#    train3[i] = my_data[1+i:4+i,a[2]:a[3]]
#    test3[i] = my_data[4+i,a[2]:a[3]]
#for i in range(0,20):
#    train4[i] = my_data[1+i:4+i,a[3]:a[4]]
#    test4[i] = my_data[4+i,a[3]:a[4]]    
#for i in range(0,20):
#    train5[i] = my_data[1+i:4+i,a[4]:a[5]]
#    test5[i] = my_data[4+i,a[4]:a[5]]    
#for i in range(0,20):
#    train6[i] = my_data[1+i:4+i,a[5]:a[6]]
#    test6[i] = my_data[4+i,a[5]:a[6]]    
#for i in range(0,20):
#    train7[i] = my_data[1+i:4+i,a[6]:a[7]]
#    test7[i] = my_data[4+i,a[6]:a[7]]
#for i in range(0,20):
#    train8[i] = my_data[1+i:4+i,a[7]:a[8]]
#    test8[i] = my_data[4+i,a[7]:a[8]]
#for i in range(0,20):
#    train9[i] = my_data[1+i:4+i,a[8]:a[9]]
#    test9[i] = my_data[4+i,a[8]:a[9]]    
print(x_dict[1][19])