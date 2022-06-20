import numpy as np
import nltk
from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
nltk.download('averaged_perceptron_tagger')


def text_tragger(data):

    assert type(data) == str, f'Expected type {str}. Received {type(data)}.'

    n_upper = 0
    n_lower = 0
    n_alpha = 0
    n_digits = 0
    n_spaces = 0
    n_numeric = 0
    n_special = 0
    number = 0
    special_chars = {'&': 0, '@': 1, '#': 2, '(': 3, ')': 4, '-': 5, '+': 6,
                     '=': 7, '*': 8, '%': 9, '.': 10, ',': 11, '\\': 12, '/': 13,
                     '|': 14, ':': 15}

    special_chars_arr = np.zeros(shape=len(special_chars))

    for char in data:

        if char.islower():
            n_lower += 1

        if char.isupper():
            n_upper += 1

        if char.isspace():
            n_spaces += 1

        if char.isalpha():
            n_alpha += 1

        if char.isnumeric():
            n_numeric += 1

        if char in special_chars.keys():
            char_idx = special_chars[char]
            special_chars_arr[char_idx] += 1

    for word in data.split():

        try:
            number = int(word)
            n_digits += 1
        except:
            pass

        if n_digits == 0:
            try:
                number = float(word)
                n_digits += 1
            except:
                pass

    features = []

    features.append([n_lower, n_upper, n_spaces, n_alpha, n_numeric, n_digits])

    features = np.array(features)

    features = np.append(features, np.array(special_chars_arr))

    return features
