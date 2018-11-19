#!/usr/bin/env python

import sys

records = []
for line in sys.stdin:
    fields = line.split('\t')
    records.append(fields[1].replace('\n', '') + '\t' + fields[0])

for i in range(10):
    print(records[len(records) - i - 1])
