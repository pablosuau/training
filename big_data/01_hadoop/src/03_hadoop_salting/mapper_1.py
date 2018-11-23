#!/usr/bin/env python

import sys

for line in sys.stdin:
    fields = line.split(' ')
    
    for word in fields:
        for letter in word.lower():
            if letter >= 'a' and letter <= 'z':
                print(letter + '\t' + str(len(word)))
