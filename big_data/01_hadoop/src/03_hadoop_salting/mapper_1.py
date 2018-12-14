#!/usr/bin/env python

import sys

# Letter e's frequency is the highest by far - applying salting
SPLIT = 2
count = 0

for line in sys.stdin:
    fields = line.split(' ')
    
    for word in fields:
        for letter in word.lower():
            if letter >= 'a' and letter <= 'z':
                if letter == 'e':
                   letter = letter + '_' + str(count % SPLIT)
                   count = count + 1
                print(letter + '\t' + str(len(word)))
