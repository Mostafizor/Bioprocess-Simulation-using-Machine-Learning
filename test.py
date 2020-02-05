import numpy as np

l = [1,2,3,4,5]
m = np.array(l)

for i, k in enumerate(sorted(m, reverse=True)):
    print(i, k)