from os.path import join
from os import walk
files = []
finfo = []
for (dirpath, dirname, filenames) in walk('./'):
    files.extend([join(dirpath,fl) for fl in filenames if fl.endswith('.m')])
for fl in files:
    with open(fl) as f:
        finfo.append((fl,len(f.readlines())))
from pandas import DataFrame
df = DataFrame(finfo, columns=['file', 'len'])
print ('count:', len(files))
print ('mean:', df.len.mean())
print ('sum:', df.len.sum())