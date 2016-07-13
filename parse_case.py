def process(line):
    wfs = []
    for item in line.strip().split('&'):
        data = item.split('=')
        if len(data) == 1 and len(data[0])>2:
#print '--->\t' + data[0].split(':')[1]
            print data[0].split(':')[1]
            continue
        if data[0] == 'wf':
            t = data[1].split('|')
            wfs.append( '%s;%d;%s' % (t[0].replace(':',''),abs(int(t[1])),t[2]))
    if len(wfs) > 0:
        print '|'.join(wfs)

import sys
def main():
    for line in sys.stdin:
        process(line)
if __name__ == '__main__':
    main()
            

            

