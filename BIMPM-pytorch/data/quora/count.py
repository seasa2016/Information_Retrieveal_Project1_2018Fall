import sys
import re
import string

def remove(line):
	if(line is None):
		return 
	line = re.sub('['+string.punctuation+']', ' ', line)
	line = re.sub('  ', ' ', line)
	return line.lower()

count = set()

with open(sys.argv[1]) as f:
    for line in f:
        for word in remove(line).strip().split():
            count.add(word)

print(len(count))