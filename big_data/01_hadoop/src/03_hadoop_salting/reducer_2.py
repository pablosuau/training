#!/usr/bin/env python

import sys

record = sys.stdin.readline()
letter, value = record.split('\t')
cma, n = value.split('_') 
cma, n = float(cma), float(n)

for line in sys.stdin:
    new_letter, value = line.split('\t')
    if new_letter == letter: 
       new_cma, new_n = value.split('_')
       new_cma, new_n = float(new_cma), float(new_n)
       cma = (cma * n + new_cma * new_n) / (n + new_n)
       n = n + new_n
    else:
       print(letter + '\t' + str(cma))
       letter = new_letter
       cma, n = value.split('_') 
       cma, n = float(cma), float(n)

print(letter + '\t' + str(cma))
