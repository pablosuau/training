#!/usr/bin/env python

import sys

current_url = None
current_count = 0
for line in sys.stdin:
    url = str(line).split('\t')[0]
    if not current_url:
        current_url = url
    if url == current_url:
       current_count = current_count + 1
    else:
       print(current_url + '\t' + str(current_count)) 
       current_url = url
       current_count = 1

print(current_url + '\t' + str(current_count))
