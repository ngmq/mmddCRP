import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(0)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    
data = np.loadtxt("data.csv", delimiter=",")
print data.shape

N = data.shape[0]
x = data[:, 0]
y = data[:, 1]

colors = np.zeros(N, dtype=np.str)
for i in range(0, 40):
    colors[i] = 'red'
for i in range(40, N):
    colors[i] = 'blue'
colors[49] = 'red'

#plt.scatter(x, y, c='red', alpha=0.5)
#plt.show()

#checkout https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
#c=numpy.random.rand(3,1)

arr_colors = np.linspace(0, 1, 21)
rb_colors = cm.rainbow(np.linspace(0, 1, 21))
#print "rb_colors = ", rb_colors

#print arr_colors
rcolors = N * [[0., 0., 0., 0.]]
rcolors2 = N * [0.]
# clustering_0000
# Mikko_clustering_0049
# Mikkko_DontcareL_clustering_0000
with open("clustering_0000.csv", "r") as f:
    cnt = 0
    for line in f:
        #rcolors2 = N * [0.]
        arr = np.fromstring(line, dtype=int, sep=',')
        #print arr
        #k =  #np.random.rand()
        for i in arr:
            rcolors[i] = rb_colors[cnt]
            rcolors2[i] = arr_colors[cnt+1]
        cnt += 1
        #rcolors = [k] * len(arr)
        #print "cnt = ", cnt
        #print "k = ", k
        #print "rcolors = ", rcolors
        #plt.scatter(x, y, c=rcolors2)
        #plt.show()
    print("Number of clusters = {}".format(cnt))
        
#print rcolors
plt.scatter(x, y, c=rcolors2)
plt.show()
