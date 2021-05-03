from matdict import loadmat
# import utils_ZuCo as uz
import os
import pickle
datadir = "D:/NLP Datasets"
dirlist = []

for subdir in os.listdir(datadir):
    for subsubdir in os.listdir(f"{datadir}/{subdir}"):
        if subsubdir.endswith('NR'):
            dirlist.append(f"{datadir}/{subdir}/{subsubdir}")
matdicts = []
for directory in dirlist:
    print("Loading files from", directory)
    count = 0
    for matfile in os.listdir(f"{directory}/Matlab files"):
        print(str(100*count/len(os.listdir(f"{directory}/Matlab files")))+"%")
        matdicts.append(loadmat(f"{directory}/Matlab files/{matfile}"))
        count += 1

print(matdicts[0])

# print(os.listdir(datadir))

with open("./matdicts.list") as mdl:
    pickle.dump(matdicts, mdl)