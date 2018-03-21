import matplotlib
import matplotlib.pyplot as plt
import numpy as np

tc = 'data/train.context'
tq = 'data/train.question'
ts = 'data/train.span'

def plotLen(fileName, label):
    y = []
    for line in open(fileName, 'r'):
        y += [len(line.split())]
    print(len(y))
    print(np.mean(y), np.max(y))
    plt.title(label)
    plt.xlabel('Length in number of words')
    plt.ylabel('Count')
    y,binEdges=np.histogram(y,bins=50)
    for i in range(len(y)):
        print((binEdges[i], y[i]), end = ' ')
    
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    plt.plot(bincenters,y,'-')
#     plt.hist(y, bins=25)
    return plt

plt = plotLen(tc, 'Distribution of train context lengths')
plt.show()
plt.clf()
plt = plotLen(tq, 'Distribution of train question lengths')
plt.show()
plt.clf()

# Plot spans
y = []
for line in open(ts, 'r'):
    l = line[:-1].split()
    y += [int(l[1]) - int(l[0])]
print(len(y))
plt.title('djkfskjdfn')
plt.xlabel('Length in number of words')
plt.ylabel('Count')
y,binEdges=np.histogram(y,bins=50)
for i in range(len(y)):
    print((binEdges[i], y[i]), end = ' ')

bincenters = 0.5*(binEdges[1:]+binEdges[:-1])

plt.plot(bincenters,y,'-')
plt.show()