import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(0)

# Have colormaps separated into categories:
# http://matplotlib.org/examples/color/colormaps_reference.html
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]



def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    
#data = np.loadtxt("data.csv", delimiter=",")
#data = np.loadtxt("jain.txt", delimiter=",")
data = np.loadtxt("aggregation.txt", delimiter=",")
#data = np.loadtxt("flame.txt", delimiter=",")

print data.shape

N = data.shape[0]
x = data[:, 0]
y = data[:, 1]

x = (x - np.min(x)) / (np.max(x) - np.min(x))
y = (y - np.min(y)) / (np.max(y) - np.min(y))
x = 2.0 * x - 1.0
y = 2.0 * y - 1.0
x -= np.mean(x)
y -= np.mean(y)

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
#rcolors = N * [[0., 0., 0., 0.]]
rcolors = N * [0]
rcolors2 = N * [0.]
# clustering_0000.csv
# Mikko_clustering_0049
# Mikkko_DontcareL_clustering_0000
# jain_groundtruth.csv
# aggregation_groundtruth.csv
# flame_groundtruth.csv
with open("clustering_0000.csv", "r") as f:
    cnt = 0
    for line in f:
        rcolors = N * [0.]
        arr = np.fromstring(line, dtype=int, sep=',')
        #print arr
        #k =  #np.random.rand()
        print "cnt = ", cnt
        for i in arr:
            rcolors[i] = arr_colors[cnt + 1]
            rcolors2[i] = arr_colors[cnt]
        cnt += 1
        #rcolors = [k] * len(arr)
        #print "cnt = ", cnt
        #print "k = ", k
        #print "rcolors = ", rcolors
        #plt.scatter(x, y, c=rcolors)
        #plt.show()
    print("Number of clusters = {}".format(cnt))
        
#print rcolors
plt.grid(True)
plt.scatter(x, y, c=rcolors2, cmap=plt.get_cmap("jet"))
plt.axes().set_aspect('equal', 'datalim')
plt.show()
