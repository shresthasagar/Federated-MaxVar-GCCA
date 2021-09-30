import numpy as np
import scipy.io as sio

fo = open('log_real_sgd_full_res_eval.txt')
sgd_obj = [fo.readline() for i in range(5500)]


sgd_obj_val = []
for line in sgd_obj:
    if ", obj:" in line:
        sgd_obj_val.append(line)

a = sgd_obj_val[49].split('"')
print(len(a))
print(a)

obj = [float(item.split('"')[7]) for item in sgd_obj_val]

# nn_freq = [float(item.split('"')[-2]) if len(item.split('"'))>9 for item in sgd_obj_val]
aroc = []
nn_freq = []
for item in sgd_obj_val:
    if len(item.split('"'))>9:
        aroc.append(float(item.split('"')[11]))
        nn_freq.append(float(item.split('"')[-2]))

# a = np.array(a)
sio.savemat( '../data/simulation_outputs/real_data_sgd_full_res_output.mat', {'obj':obj, 'aroc': aroc, 'nn_freq': nn_freq})
