import sys
#import util
import math

class grid_util:
    xbit_size = 16

    @staticmethod
    def get_gxy_xy(x,y,grid_size,precise):
        grid_size = grid_size / precise
        x = int(x / grid_size)
        y = int(y / grid_size)
        return [x,y]

    @staticmethod
    def get_grid2xy_dist(xy,gridxy,grid_size = 150,precise = 5):
        xy = grid_util.get_grid_xy(xy[0],xy[1],grid_size,precise)
        (distx,disty) = (xy[0] - gridxy[0], xy[1] - gridxy[1])
        return math.sqrt(distx * distx + disty * disty) * grid_size

    @staticmethod
    def get_orixy_by_grid(gridid,grid_size,precise):
        grid_size = grid_size / precise
        y = gridid >> grid_util.xbit_size
        x = gridid & ((1<<grid_util.xbit_size) - 1)
        return ((x+0.5)*grid_size ,(y+0.5)*grid_size)

    @staticmethod
    def get_grid_by_gxy(x,y):
        f = (1<<grid_util.xbit_size) - 1
        return int(x) & f + ((int(y) & f) << grid_util.xbit_size)
        
    @staticmethod
    def get_grid_ids(x,y,grid_size = 150,precise = 5):
        (x,y) = grid_util.get_gxy_xy(x,y,grid_size,precise)
        mid = int(precise / 2)
    
        f = (1<<grid_util.xbit_size) - 1
#print f,x &f , y&f,int(x&f)+int((y&f)<<grid_util.xbit_size)
#       print [ (x  & f + ((y & f) << grid_util.xbit_size)) for i in range(precise) for j in range(precise) ]
        return [ (int((x + i - mid) & f) + (((y+j-mid) & f) << grid_util.xbit_size)) for i in range(precise) for j in range(precise) ]
    
    
    @staticmethod
    def get_grid_with_center(x,y,grid_size = 150,precise = 5):
        (x,y) = grid_util.get_gxy_xy(x,y,grid_size,precise)

        f = (1<<grid_util.xbit_size) - 1
        return int(x & f)+ int((y & f) << grid_util.xbit_size)
    def xy_in_grid(gridid,x,y):
        return True

import datetime

def is_night(dt):
    if dt.hour >= 20 or dt.hour <=8:
        return 1
    return 0

def is_weekend(dt):
    return int(dt.weekday() >= 6)

def test():
    for line in sys.stdin:
        try:
            g = line.strip().split('\t')
#(x,y) = util.coordtrans("wgs84","bd09mc",float(g[2]),float(g[3]))
            (x,y)=(float(g[0]),float(g[1]))
#dt = datetime.datetime.fromtimestamp(long(g[1]))
            dt = datetime.datetime.now()
            xy = grid_util.get_grid_xy(x,y,500,5)
            gridid = grid_util.get_grid_by_gxy(xy[0],xy[1])
            print '%s\t%d\t%d\t%d\t%d' % (gridid,xy[0],xy[1],is_night(dt),is_weekend(dt))
        except :
            sys.stderr.write(repr(sys.exc_info())+'\n')
            pass
def test2():    
    test_x = 12949405
    test_y = 4845437
    print grid_util.get_grid_ids(test_x,test_y,3000,3)
    id = grid_util.get_grid_with_center(test_x,test_y,3000,3)
    print test_x,test_y,id
    print grid_util.get_orixy_by_grid(id,3000,3)

if __name__ == '__main__':
    test2()
