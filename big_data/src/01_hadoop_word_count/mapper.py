import sys

FILTER = ['\n', 
          '\r', 
           '.', 
           ',', 
           ';', 
           ':', 
           '(',
           '[',
            ']',
            ')',
            '\'s', 
            '"', 
            '!', 
            '?']

for line in sys.stdin:
    for word in line.split(' '):
        for f in FILTER:
            word = word.replace(f, '')
        if word and not word.startswith('http'):
            print(word.lower() + ' 1')
