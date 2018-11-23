#!/usr/bin/env python

import sys

# Inverting the fields so a single reducer can filter out the 10 first
# records after Hadoop's sort and suffle 
for line in sys.stdin:
    fields = str(line).split('\t')
    print(fields[1].replace('\n', '') + '\t' + fields[0])
