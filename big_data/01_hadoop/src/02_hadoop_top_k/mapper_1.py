#!/usr/bin/env python

import sys

for line in sys.stdin:
    fields = line.split(' ')
    print(fields[0] + '\t1') 
