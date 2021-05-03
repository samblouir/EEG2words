import pickle

def main():
    data_file = "D:\\NLP Datasets\\word_dict.dict"
    new_dict_file = "D:\\NLP Datasets\\word_dict_new.dict"
    
    print("Loading dictionary from:", data_file)
    with open(data_file, 'rb') as wd:
        data = pickle.load(wd)

    new_dict = {}
    totallen = 0
    for idx, word in enumerate(data):
        for occurence in data[word]:
            for fix in occurence['RAW_EEG']:
                if len(fix) >= 51:
                    if word not in new_dict:
                        new_dict[word] = []
                    for trim in range(int(len(fix)/51)):
                        trimstart = trim*51
                        trimend = trimstart+51
                        new_dict[word].append(fix[trimstart:trimend])
        print(f"Word {idx+1}/{len(data)}")
        if word in new_dict:
            print("Length:", len(new_dict[word]))
            totallen += len(new_dict[word])
        else:
            print(word, "has no data.")
    print("Total number of data points:",totallen)
    print("Average number of data points per word:", totallen/len(new_dict))
    print("Saving dictionary to:", new_dict_file)
    with open(new_dict_file, 'wb') as wd:
        pickle.dump(new_dict, wd)

if __name__ == "__main__":
    main()