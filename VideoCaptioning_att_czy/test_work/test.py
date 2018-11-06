
infile = '/home/ytkj/w2v/wiki_zh_2g.txt'
out_file = '/home/ytkj/w2v/new_wiki_zh_2g.txt'

file = open(infile, 'r')
out = open(out_file, 'w')
try:
    for line in file:

            if not isinstance(line, str):
                print('isinstance')
                continue
            out.write(line)
except:
    print('except')
    pass