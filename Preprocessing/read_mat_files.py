import os
import scipy.io as io
import pickle
import numpy as np
import string
import data_loading_helpers as dh

### HOW TO READ MATLAB FILE IN PYTHON 3 ###
task = "NR"
# set correct file name
rootdir = "D:\\NLP Datasets\\EEG Eye-Tracked reading data\\task2 - NR\\Matlab files\\"

###########################################################################################
# with open("NR_TSR_word_dict.dict", 'rb') as df:
#     wordDict = pickle.load(df)
wordDict = {}

for fidx, file in enumerate(os.listdir(rootdir)):
    if file.endswith(task+".mat"):
        print(file, f"({fidx+1}/{len(os.listdir(rootdir))})")

        file_name = rootdir + file
        subject = file_name.split("s")[1].split("_")[0]

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':

            #f = h5py.File(file_name, 'r+')
            f = io.loadmat(file_name, squeeze_me=True, struct_as_record=False)
            for idx, sentence_data in enumerate(f['sentenceData']):

                rawData = sentence_data.rawData
                contentData = sentence_data.content
                omissionR = sentence_data.omissionRate
                wordData = sentence_data.word

                # number of sentences:
                # print(len(rawData))
                if idx%10 == 0:
                    print(f"{'{:.2f}'.format((idx/len(f['sentenceData']))*100)}%")

                sent = contentData

                # get omission rate
                obj_reference_omr = omissionR
                omr = np.array(obj_reference_omr)
                # print(omr)

                # get word level data
                word_data = dh.extract_word_level_data(f, wordData)

                # number of tokens
                # print(len(word_data))
                print(word_data)
                for widx in range(len(word_data)):
                    print(word_data[widx].keys())

                    #print(word_data[widx]['content'])
                    # get first fixation duration (FFD)
                    #print(word_data[widx]['FFD'])
                    # Punctuation stripping from:
                    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
                    word = word_data[widx]['content'].translate(str.maketrans('', '', string.punctuation)).lower()
                    # print(word, word_data[widx]['word_idx'])
                    if word not in wordDict:
                        wordDict[word] = []
                    occurrence = {}
                    for category in word_data[widx].keys():
                        # Skip unnecessary categories
                        if category in ['word_idx', 'content']:
                            continue
                        occurrence[category] = word_data[widx][category]
                    wordDict[word].append(occurrence)
                    # get aggregated EEG alpha features
                    #print(word_data[widx]['ALPHA_EEG'])



with open("attempt_word_dict.dict", 'wb') as wdfile:
    pickle.dump(wordDict, wdfile)

########################################################################################################
'''
# index of the array `data` is the number of sentence
data = io.loadmat(rootdir, squeeze_me=True, struct_as_record=False)['sentenceData']

# get all field names for sentence data
print(data[0]._fieldnames)
# example: print sentence
print(data[0].content)

# example: get omission rate of first sentence
omission_rate = data[0].omissionRate
print(omission_rate)

# get word level data
word_data = data[0].word

# get names of all word features
# index of the array `word_data` is the number of the word
print(word_data[0]._fieldnames)

# example: get first word
print(word_data[0].content)

# example: get number of fixations of first word
print(word_data[0].nFixations)
'''