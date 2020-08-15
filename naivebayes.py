#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Hely Modi (helymodi), Dhruva Bhavsar(dbhavsar), Aneri Shah(annishah)
#

import sys
import os
import numpy as np

def build_dict(path):
    spam, notspam = [], []
    word_dict = {}

    # Extract files for train data and test data
    for dirs, subdirs, files in os.walk(path):
        if (os.path.split(dirs)[1] == 'spam'):
            for filename in files:
                with open(os.path.join(dirs, filename), 'r', encoding="Latin-1") as f:
                    data = f.read().lower()
                    spam.append(data)
                    words = list(data.lower().split(' '))
                    for word in words:
                        if word not in word_dict:
                            word_dict[word] = 1
                        else:
                            word_dict[word] += 1

        elif (os.path.split(dirs)[1] == 'notspam'):
            for filename in files:
                with open(os.path.join(dirs, filename), 'r', encoding="Latin-1") as f:
                    data = f.read().lower()
                    notspam.append(data)
                    words = list(data.lower().split(' '))
                    for word in words:
                        if word not in word_dict:
                            word_dict[word] = 1
                        else:
                            word_dict[word] += 1

    # Build word dictionary
    remove_list = list(word_dict.keys())
    for item in remove_list:
        if item.isalpha() == False:
            del word_dict[item]
        elif len(item) <= 2:
            del word_dict[item]

    return spam, notspam, word_dict

def calc_prior(spam, notspam, word_dict):
    spam_words, notspam_words = {}, {}

    for file in spam:
        words = list(file.lower().split(' '))
        for word in words:
            if word in word_dict:
                if word not in spam_words:
                    count = 1
                    spam_words[word] = (count)
                else:
                    count = spam_words[word]
                    spam_words[word] = (count+1)

    for file in notspam:
        words = list(file.lower().split(' '))
        for word in words:
            if word in word_dict:
                if word not in notspam_words:
                    count = 1
                    notspam_words[word] = (count)
                else:
                    count = notspam_words[word]
                    notspam_words[word] = (count+1)

    # Calculate prior probability
    spam_word_count, notspam_word_count = 0, 0
    for word in spam_words:
        spam_word_count += spam_words[word]

    for word in notspam_words:
        notspam_word_count += notspam_words[word]

    # Uses smoothing factor of 1 to avoid zero probability
    alpha = 1
    for word in spam_words:
        val = spam_words[word]
        prior = (val + alpha) + (spam_word_count + alpha*len(word_dict))
        spam_words[word] = (val, prior)

    for word in notspam_words:
        val = notspam_words[word]
        prior = (val + alpha) / (notspam_word_count + alpha*len(word_dict))
        notspam_words[word] = (val, prior)

    return spam_words, notspam_words

if __name__ == '__main__':
    if (len(sys.argv) != 4):
        raise Exception("usage: ./spam.py training-directory testing-directory output-file")

    path = sys.argv[1] + '/'          #'train/'

    spam_train, notspam_train, word_dict = build_dict(path)
    p_s = len(spam_train) / (len(spam_train) + len(notspam_train))
    p_ns = 1 - p_s

    # Calculates P(W|Spam), P(W|NotSpam)
    spam_words, notspam_words = calc_prior(spam_train, notspam_train, word_dict)

    path = sys.argv[2] + '/'          #'test/'
    files, result = [], []
    for file in os.listdir(path):
        p = 1
        with open(path+file, 'r', encoding="Latin-1") as f:
            data = f.read()
            words = list(data.lower().split(' '))
            for word in words:
                if word in spam_words and word in notspam_words:
                    val1 = spam_words[word]
                    val2 = notspam_words[word]
                    p *= ((val1[1] * p_s) / ((val1[1] * p_s) + (val2[1] * p_ns)))
                elif word in spam_words:
                    val1 = spam_words[word]
                    val2 = (0,0)
                    p *= ((val1[1] * p_s) / ((val1[1] * p_s) + (val2[1] * p_ns)))
                elif word in notspam_words:
                    val1 = (0,0)
                    val2 = notspam_words[word]
                    p *= ((val1[1] * p_s) / ((val1[1] * p_s) + (val2[1] * p_ns)))
        if p >= 0.5:
            result.append(1)
            files.append((file, 'spam'))
        else:
            result.append(0)
            files.append((file, 'notspam'))


    # Save the output
    output = sys.argv[3]
    np.savetxt(output, files, fmt='%s')
    # print('Output saved to file!')