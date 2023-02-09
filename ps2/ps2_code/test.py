import numpy as np

my_list = []

for i in range(5):
    my_list.append([i, i*2])

print(my_list)

my_arr = np.vstack(my_list)
print(my_arr)