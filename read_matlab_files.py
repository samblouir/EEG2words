import os
import numpy as np
import h5py
import data_loading_helpers as dh
import pickle
import string

task = "NR"

rootdir = "D:\\NLP Datasets\\EEG Eye-Tracked reading data 2.0\\task1 - NR\\Matlab files\\"


sentences = {}

wordDict = {}

for fidx, file in enumerate(os.listdir(rootdir)):
    if file.endswith(task+".mat"):
        print(file, f"({fidx+1}/{len(os.listdir(rootdir))})")

        file_name = rootdir + file
        subject = file_name.split("s")[1].split("_")[0]

        # exclude YMH due to incomplete data because of dyslexia
        if subject != 'YMH':

            f = h5py.File(file_name, 'r+')
            sentence_data = f['sentenceData']
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']

            # number of sentences:
            # print(len(rawData))

            for idx in range(len(rawData)):
                if idx%10 == 0:
                    print(f"{'{:.2f}'.format((idx/len(rawData))*100)}%")
                obj_reference_content = contentData[idx][0]
                sent = dh.load_matlab_string(f[obj_reference_content])

                # get omission rate
                obj_reference_omr = omissionR[idx][0]
                omr = np.array(f[obj_reference_omr])
                # print(omr)

                # get word level data
                word_data = dh.extract_word_level_data(f, f[wordData[idx][0]])

                # number of tokens
                # print(len(word_data))

                for widx in range(len(word_data)):
                    # print(word_data[widx].keys())

                    #print(word_data[widx]['content'])
                    # get first fixation duration (FFD)
                    #print(word_data[widx]['FFD'])
                    # Punctuation stripping from:
                    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
                    word = word_data[widx]['content'].translate(str.maketrans('', '', string.punctuation))
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
            f.close()


with open("word_dict.dict", 'wb') as wdfile:
    pickle.dump(wordDict, wdfile)