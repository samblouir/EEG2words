import os

from shared_functions import load_mat

os.system(f"clear")
print("hello!")

raw_0 = "data_v1/task1-SR/Raw data/ZAB/ZAB_SNR6_EEG.mat"
load_mat(raw_0)