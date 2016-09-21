# import os
import numpy as np
import matplotlib.pyplot as plt
import random

# cluster_dp_GPU = "./cluster_dp_GPU"
# os.system(cluster_dp_GPU)
result_file = raw_input("enter the result file name:")
location = []
input_lable = []
for line in open("dataset_2D.txt", "r"):
    items = line.strip("\n").split(",")
    input_lable.append(int(items.pop()))
    tmp = []
    for item in items:
        tmp.append(float(item))
    location.append(tmp)
location = np.array(location)
input_lable = np.array(input_lable)
length = len(location)
print "data input complete"
result_lable = []
for line in open(result_file, "r"):
    items = line.strip("\n").split(",")
    result_lable.append(int(items.pop()))
print "result read complete"
R = range(256)
random.shuffle(R)
random.shuffle(R)
R = np.array(R) / 255.0
G = range(256)
random.shuffle(G)
random.shuffle(G)
G = np.array(G) / 255
B = range(256)
random.shuffle(B)
random.shuffle(B)
B = np.array(B) / 255.0
colors = []
for i in range(256):
    colors.append((R[i], G[i], B[i]))

plt.figure()
for i in range(length):
    index = input_lable[i]
    plt.plot(location[i][0], location[i][1], color=(R[index*5],G[index*15],B[index*20]), marker='.')
plt.xlabel('x'), plt.ylabel('y')
plt.show()
# plt.close()
plt.figure()
for i in range(length):
    index = result_lable[i]
    plt.plot(location[i][0], location[i][1], color=(R[index*5],G[index*15],B[index*20]), marker='.')
plt.xlabel('x'), plt.ylabel('y')
plt.show()
