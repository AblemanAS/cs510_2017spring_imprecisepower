import matplotlib.pyplot as plt
import numpy as np

DATA_ORI_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\BF hotspot\\hotspot\\output.out'
DATA_COMP_PATH = 'C:\\Users\\zkalf\\Desktop\\Result\\AF hotspot\\hotspot\\output.out'

# original
fp = open(DATA_ORI_PATH, 'r')
data = fp.readlines()
for i in range(len(data)):
    data[i] = float(data[i].split('\t')[1])
fp.close()
heatmap = np.array(data).reshape((512, 512))

plt.imshow(heatmap, cmap='hot', interpolation='nearest')
cb = plt.colorbar()
cb.set_label('Temperature (Kelvin)')
plt.title('Original')
plt.figure()

plt.plot(data, color='red')
plt.title('Original')
plt.figure()

# store original data
ori_data = data

# compare
fp = open(DATA_COMP_PATH, 'r')
data = fp.readlines()
for i in range(len(data)):
    data[i] = float(data[i].split('\t')[1])
fp.close()
heatmap = np.array(data).reshape((512, 512))

# calc MSE
diff = np.array(data) - np.array(ori_data)
diff = diff.reshape((1, len(diff)))
MSE = np.dot(diff, diff.T)[0][0] / len(data)
compStr = ' MSE={:.6f}'.format(MSE) + ' (var=' + '{:.2f}'.format(np.var(data)) + ')'

plt.imshow(heatmap, cmap='hot', interpolation='nearest')
cb = plt.colorbar()
cb.set_label('Temperature (Kelvin)')
plt.title('Compare' + compStr)
plt.figure()

plt.plot(data, color='red')
plt.title('Compare' + compStr)
plt.show()
