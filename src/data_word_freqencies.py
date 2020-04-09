from nltk.tokenize import sent_tokenize, word_tokenize
import json
import nltk


def calculate_frequencies(file_path, out_file_name):
    sample = open(file_path, "r", encoding='utf-8')
    s = sample.read()
    f = s.replace("\n", " ")
    data = []
    words = []
    for i in sent_tokenize(f):
        temp = []
        for j in word_tokenize(i):
            temp.append(j.lower())
            words.append(j.lower())
        data.append(temp)
    num_words = float(len(words))
    fdist = nltk.FreqDist(words)
    fdist['**TOTAL**'] = float(num_words)
    f = open('frequencies/{}_word_freq.json'.format(out_file_name), 'w')
    json.dump(fdist, f)
    f.close()
