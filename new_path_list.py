from shared_functions import timerD
from __main__ import data
import os, random
import numpy as np

# dict_keys(['RAW_EEG', 'RAW_ET', 'FFD', 'GD', 'GPT', 'TRT', 'SFD', 'nFix', 'ALPHA_EEG', 'BETA_EEG', 'GAMMA_EEG', 'THETA_EEG'])

# This runs safely in a new thread
os.system("clear")

print(f"len(data): {len(data)} \n ")

x = random.randint(0, len(data))
x_fmt = f"{x:4d}"
name = data[x][0]

print(f"  data[{x_fmt}][0]: \"{data[x][0]}\"")
print(f"    len(data[{x_fmt}][1]): {len(data[x][1])}")

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################


keys = data[x][1][0].keys()
new_dict = {k: [k] for k in keys}
new_list = [[k] for k in keys]
print(keys)

# for _i in range(0, len(data), 1000):
for _i in range(16, 17):
    name = data[_i][0]
    values = data[_i][1]
    for i in range(len(values)):
        print()
        for count, key in enumerate(keys):
            # if count > 0:
            #     break
            # if key == "RAW_EEG" or key == ""

            ck = values[i]
            new_list[count].append(ck)

            # print(f"    len(ck[{key}]): ", end='', flush=True)
            try:
                print(f"    len(data[{name}][1][{i}][{key}]): {len(data[_i][1][i][key])}")
            except:
                # print()
                pass
                # print(f"    data[{name}][1][{i}][{key}]: {data[_i][1][i][key]}")


# for i in range(len(data[x][1])):
#     curr = data[x][1][i]
#     msg = f"  #{i}: {len(curr)}"
#     print(msg)
#
#     for count, key in enumerate(keys):
#
#         ck = curr[key]
#         new_list[count].append(ck)
#
#         print(f"    len(curr[{key}]): ", end='', flush=True)
#         try:
#             print(f"    len(curr[{key}]): {len(curr[key])}")
#         except:
#             print(f"    curr[{key}]: {curr[key]}")

# for e in new_list:
#     print(f"        {name}: {e[0]}: {len(e[1:])}")

# ourself = data[x]
# ourself = np.array(ourself, dtype=object)
# print(ourself)
# np.save("ourself", ourself, fix_imports=False)
#
# new_list = np.array(new_list, dtype=object)
# print(new_list)
# np.save("new_list", new_list, fix_imports=False)
# print(f"\n")

print()
timerD(f"Finished printing \"{name}\".")
