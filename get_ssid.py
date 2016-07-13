import redis
import sys

def conn():
#    return redis.StrictRedis(host = 'hz01-maprd-da01.hz01.baidu.com',port = 9302,db = 0)
    return redis.StrictRedis(host = '127.0.0.1',port = 16179,db = 0)

class fetch_data:
    def __init__(self):
        self.pipe = conn().pipeline()
        self.conn = conn()
        self.data = []
    def get_data(self,t,idx):
        self.data.append(t)
        mac = 0
        try:
            mac = long(t[idx],base=16)
        except:
            pass
        self.pipe.hget(mac,1)
        if len(self.data) > 5000:
            self.print_result()
            self.data = []

    def get_ssid(self,mac):
        return self.conn.hget(mac,1)

    def print_result(self)  :
        r = self.pipe.execute()
        for i in range(len(self.data)):
            print '%s\t%s' % ('\t'.join(self.data[i]),r[i])


def main():
    fd = fetch_data()
#    r = conn()
    if len(sys.argv) > 1:
        i = int(sys.argv[1])
    else:
        i = 0
    for line in sys.stdin:
        g = line.strip().split()
#        ssid = r.hget(long(g[i],base=16),0)
#        print '%s\t%s' % (line.strip(),str(ssid))
        fd.get_data(g,i) 
    fd.print_result()
if __name__ == '__main__'        :
    main()
    
