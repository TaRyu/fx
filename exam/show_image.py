import numpy as np
import matplotlib.pyplot as plt


FILE_PREX = '../../../data/fx'

fig, ax = plt.subplots()

n = np.load('%s/Fs/EURUSD_1.npy' % FILE_PREX)[0]

n_range = np.max(n) - np.min(n)
image = np.array([(n[i] - np.min(n)) / n_range for i in range(len(n))])
ax.imshow(image.reshape(24, 24), cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('dropped spines')

# Move left and bottom spines outward by 10 points
ax.spines['left'].set_position(('outward', 24))
ax.spines['bottom'].set_position(('outward', 24))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16,
        }

plt.xlabel('24 hours', fontdict=font)
plt.ylabel('24 days', fontdict=font)

plt.show()
