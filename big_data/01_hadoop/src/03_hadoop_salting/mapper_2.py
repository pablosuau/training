#!/usr/bin/env python

import sys

# In this mapper we remove the suffix added by salting

for line in sys.stdin:
    letter, value = line.split('\t')

    if letter[0] == 'e':
       letter = 'e'

    print(letter + '\t' + value.replace('\n', ''))
