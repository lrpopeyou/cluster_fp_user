class Grid:
    def __init__(self,grid_size=150,align = 60):
        self.grid_size = 150
        self.align = 50
        self.idbit = 40
        self.headbit = 24

        self.header = (grid_size << (self.headbit/2)) | align

    def get_gridids_with_align(self,x,y):
        ids = set()
        for xi in (x,x - self.align,x + self.align):
            for yi in (y,y - self.align,y + self.align):
#   print 'debug:',xi,yi
                ids.add(self.get_gridid(xi,yi))
        return list(ids)    
    
    def get_gridid(self,x,y):
        x = int(x / self.grid_size)
        y = int(y / self.grid_size)
#       print x,y
        f = (1<<(self.idbit / 2))  -1
        return (self.header << self.idbit) \
            | ((x & f) << (self.idbit / 2)) \
            | (y & f)
        
    def reset_with_id(self,gridid):
        header = gridid >> self.idbit
        self.grid_size = header >> (self.headbit /2)
        self.align = header & (1 << (self.headbit /2 ))
 
    def get_center_xy(self,gridid):
        header = gridid >> self.idbit
        grid_size = header >> (self.headbit /2)
        xy = gridid & ((1<<self.idbit) -1) 
        xi = xy >> (self.idbit /2)
        yi = xy & ((1 << self.idbit /2) -1)
        return (xi+0.5) * grid_size,(yi+0.5) * grid_size

    def is_ingrid(self,gridid,x,y):
        return gridid == self.get_gridid(x,y)


def test(x,y):
    G = Grid()
    ids = G.get_gridids_with_align(x,y)
    print 'test--->', int(x),int(y)
    print ids
    for id in ids:
        print G.get_center_xy(id),G.is_ingrid(id,x,y)

if __name__ == '__main__':
    (x,y) = (12948581.938752,4843228.38910)
    test(x,y)
    (x,y) = (12948500.938752,4843200.38910)
    test(x,y)
    
        
