#!/usr/bin/env python

import sys

sources = []
types = []

for line in sys.stdin:
    fields = line.encode('utf-8').strip().decode('utf-8').split(' ')
    print(fields[5].replace('"', '') + '_' + fields[0] + '\t1') 
