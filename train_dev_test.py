import pickle


def main():

    datafile = "data/WD/word_dict_new.dict"
    trainfile = "data/WD/train_dict.dict"
    devfile = "data/WD/dev_dict.dict"
    testfile = "data/WD/test_dict.dict"
    print("Loading data file from:", datafile)
    with open(datafile, 'rb') as wd:
        data = pickle.load(wd)

    # Remove words with only 1 eeg sample
    remove = [word for word in data if len(data[word]) == 1]
    if len(remove) != 0:
        for word in remove:
            data.pop(word)

    traindict = {}
    devdict = {}
    testdict = {}
    print("Divying out data...")
    trainsize = 0
    devsize = 0
    testsize = 0
    for word in data:
        for idx, eeg in enumerate(data[word]):
            if idx % 4 < 3:
                if idx % 4 < 2:
                    # 50% train
                    if word not in traindict:
                        traindict[word] = []
                    traindict[word].append(eeg)
                    trainsize += 1
                else:
                    # 25% dev
                    if word not in devdict:
                        devdict[word] = []
                    devdict[word].append(eeg)
                    devsize += 1
            else:
                # 25% test
                if word not in testdict:
                    testdict[word] = []
                testdict[word].append(eeg)
                testsize += 1

    print("Train size:", trainsize)
    print("Dev size:", devsize)
    print("Test size:", testsize)

    print("Saving train dictionary...")
    with open(trainfile, 'wb') as tf:
        pickle.dump(traindict, tf)
    print("Saving dev dictionary...")
    with open(devfile, 'wb') as df:
        pickle.dump(devdict, df)
    print("Saving test dictionary...")
    with open(testfile, 'wb') as tf:
        pickle.dump(testdict, tf)

if __name__ == "__main__":
    main()