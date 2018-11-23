#!/usr/bin/env python

import sys

record = sys.stdin.readline()
word = record.split('\t')[0]
count = 1

for line in sys.stdin:
    new_word = line.split('\t')[0]
    if new_word == word:
       count = count + 1
    else:
       print(word + '\t' + str(count))
       word = new_word
       count = 1

print(word + '\t' + str(count))
