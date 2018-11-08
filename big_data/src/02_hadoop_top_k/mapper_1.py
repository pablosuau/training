#!/usr/bin/env python

import sys
import io

input_stream = io.TextIOWrapper(sys.stdin.buffer, 
                                encoding = 'utf-8',
                                errors = 'replace')

for line in input_stream:
    fields = str(line).split(' ')
    print(fields[0] + '\t1') 
