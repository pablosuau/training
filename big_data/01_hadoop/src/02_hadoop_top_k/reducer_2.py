#!/usr/bin/env python

import sys

i = 0
for line in sys.stdin:
    fields = line.split('\t')
    if i < 10:
       print(fields[1].replace('\n', '') + '\t' + fields[0])
    i = i + 1

