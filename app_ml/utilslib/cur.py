import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image, ImageDraw
from math import sqrt
import random

from utils import DATA_DIR
import os
import numpy as np
import matplotlib.pylab as pl
from sklearn.neighbors  import KNeighborsClassifier as KNN
from sklearn.cross_validation import cross_val_score

g_train = None
g_test = None

########################################################################
##linregPolyVs

########################################################################
##knn
# def load_data():
#     global g_train
#     global g_test

#     train_file = os.path.join(DATA_DIR, 'knnClassify3cTrain.txt')
#     test_file = os.path.join(DATA_DIR, 'knnClassify3cTest.txt')
#     train = np.loadtxt(train_file, dtype=[('x_train', ('f8', 2)),
#                                         ('y_train', ('f8', 1))])
#     g_train = train
#     test = np.loadtxt(test_file, dtype=[('x_test', ('f8', 2)),
#                                         ('y_test', ('f8', 1))])
#     g_test = test

#     return train['x_train'], train['y_train'], test['x_test'], test['y_test']

# x_train, y_train, x_test, y_test = load_data()
# pl.figure()
# y_unique = np.unique(y_train)
# markers = '*x+'
# colors = 'bgr'
# for i in range(len(y_unique)):
#     pl.scatter(x_train[y_train==y_unique[i], 0],
#                 x_train[y_train==y_unique[i], 1],
#                 marker=markers[i],
#                 c=colors[i])
# pl.savefig('knnClassifyDemo_1.png')

#plot test fig
# pl.figure()
# for i in range(len(y_unique)):
#     pl.scatter(x_test[y_test==y_unique[i], 0],
#                 x_test[y_test==y_unique[i], 1],
#                 marker=markers[i],
#                 c=colors[i])
# pl.savefig('knnClassifyDemo_2.png')

# def get_xy():
#     x = np.linspace(np.min(x_test[:, 0]), np.max(x_test[:, 0]), 200)
#     y = np.linspace(np.min(x_test[:, 1]), np.max(x_test[:, 1]), 200)
#     xx, yy = np.meshgrid(x, y)
#     xy = np.c_[xx.ravel(), yy.ravel()]
#     print("shape of xy: ", xy.shape)

#     return xy

# xy = get_xy()

# for k in [1, 5, 10]:
#     knn = KNN(n_neighbors = k)
#     knn.fit(x_train, y_train)
#     pl.figure()
#     y_predicted = knn.predict(xy)
#     pl.pcolormesh(y_predicted.reshape(200,200))
#     pl.title('k=%s' %(k))
#     pl.savefig('knnCalssifyDemo_k%s.png' % (k))

#plot train err and test err with different k
# ks = [1, 5, 10, 20, 50, 100, 120]
# train_errs = []
# test_errs = []
# for k in ks:
#     knn = KNN(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     train_errs.append(1 - knn.score(x_train, y_train))
#     test_errs.append(1 - knn.score(x_test, y_test))
# pl.figure()
# pl.plot(ks, train_errs, 'bs:', label='train')
# pl.plot(ks, test_errs, 'rx-', label='test')
# pl.legend()
# pl.xlabel('k')
# pl.ylabel('misclassification rate')
# pl.savefig('knnClassifyDemo_4.png')

#cross_validate
# scores = []
# for k in ks:
#     knn = KNN(n_neighbors=k)
#     score = cross_val_score(knn, x_train, y_train, cv=5)
#     scores.append(1 - score.mean())

# pl.figure()
# pl.plot(ks, scores, 'ko-')
# pl.xlabel('k')
# pl.ylabel('misclassification rate')
# pl.title('5-fold cross validateion, n-train = 200')

#draw hot-map to show the probability of different class
# knn = KNN(n_neighbors = 10)
# knn.fit(x_train, y_train)
# xy_predict = knn.predict_proba(xy)
# levels = np.arange(0, 1.01, 0.1)
# for i in range(3):
#     pl.figure()
#     pl.contourf(xy_predict[:, i].ravel().reshape(200, 200), levels)
#     pl.colorbar()
#     pl.title('p(y=%s | data, k=10)' % (i))
#     pl.savefig('knnClassifyDemo_hotmap_%s.png' % (i))
# pl.show()

########################################################################
## pci data
## /Users/linked0/kg/data

# def readfile(filename):
#     lines = [line for line in open(filename)]

#   # First line is the column titles
#     colnames = lines[0].strip().split('\t')[1:]
#     rownames = []
#     data = []
#     for line in lines[1:]:
#         p = line.strip().split('\t')
#     # First column in each row is the rowname
#         rownames.append(p[0])
#     # The data for this row is the remainder of the row
#         data.append([float(x) for x in p[1:]])
#     return (rownames, colnames, data)


# def pearson(v1, v2):
#   # Simple sums
#     sum1 = sum(v1)
#     sum2 = sum(v2)

#   # Sums of the squares
#     sum1Sq = sum([pow(v, 2) for v in v1])
#     sum2Sq = sum([pow(v, 2) for v in v2])

#   # Sum of the products
#     pSum = sum([v1[i] * v2[i] for i in range(len(v1))])

#   # Calculate r (Pearson score)
#     num = pSum - sum1 * sum2 / len(v1)
#     den = sqrt((sum1Sq - pow(sum1, 2) / len(v1)) * (sum2Sq - pow(sum2, 2)
#                / len(v1)))
#     if den == 0:
#         return 0

#     return 1.0 - num / den


# class bicluster:

#     def __init__(
#         self,
#         vec,
#         left=None,
#         right=None,
#         distance=0.0,
#         id=None,
#         ):
#         self.left = left
#         self.right = right
#         self.vec = vec
#         self.id = id
#         self.distance = distance


# def hcluster(rows, distance=pearson):
#     distances = {}
#     currentclustid = -1

#   # Clusters are initially just the rows
#     clust = [bicluster(rows[i], id=i) for i in range(len(rows))]

#     while len(clust) > 1:
#         lowestpair = (0, 1)
#         closest = distance(clust[0].vec, clust[1].vec)

#     # loop through every pair looking for the smallest distance
#         for i in range(len(clust)):
#             for j in range(i + 1, len(clust)):
#         # distances is the cache of distance calculations
#                 if (clust[i].id, clust[j].id) not in distances:
#                     distances[(clust[i].id, clust[j].id)] = \
#                         distance(clust[i].vec, clust[j].vec)

#                 d = distances[(clust[i].id, clust[j].id)]

#                 if d < closest:
#                     closest = d
#                     lowestpair = (i, j)

#     # calculate the average of the two clusters
#         mergevec = [(clust[lowestpair[0]].vec[i] + clust[lowestpair[1]].vec[i])
#                     / 2.0 for i in range(len(clust[0].vec))]

#     # create the new cluster
#         newcluster = bicluster(mergevec, left=clust[lowestpair[0]],
#                                right=clust[lowestpair[1]], distance=closest,
#                                id=currentclustid)

#     # cluster ids that weren't in the original set are negative
#         currentclustid -= 1
#         del clust[lowestpair[1]]
#         del clust[lowestpair[0]]
#         clust.append(newcluster)

#     return clust[0]


# def printclust(clust, labels=None, n=0):
#   # indent to make a hierarchy layout
#     for i in range(n):
#         print(' '),
#     if clust.id < 0:
#     # negative id means that this is branch
#         print('-')
#     else:
#     # positive id means that this is an endpoint
#         if labels == None:
#             print(clust.id)
#         else:
#             print(labels[clust.id])

#   # now print the right and left branches
#     if clust.left != None:
#         printclust(clust.left, labels=labels, n=n + 1)
#     if clust.right != None:
#         printclust(clust.right, labels=labels, n=n + 1)


# def getheight(clust):
#   # Is this an endpoint? Then the height is just 1
#     if clust.left == None and clust.right == None:
#         return 1

#   # Otherwise the height is the same of the heights of
#   # each branch
#     return getheight(clust.left) + getheight(clust.right)


# def getdepth(clust):
#   # The distance of an endpoint is 0.0
#     if clust.left == None and clust.right == None:
#         return 0

#   # The distance of a branch is the greater of its two sides
#   # plus its own distance
#     return max(getdepth(clust.left), getdepth(clust.right)) + clust.distance


# def drawdendrogram(clust, labels, jpeg='clusters.jpg'):
#   # height and width
#     h = getheight(clust) * 20
#     w = 1200
#     depth = getdepth(clust)

#   # width is fixed, so scale distances accordingly
#     scaling = float(w - 150) / depth

#   # Create a new image with a white background
#     img = Image.new('RGB', (w, h), (255, 255, 255))
#     draw = ImageDraw.Draw(img)

#     draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))

#   # Draw the first node
#     drawnode(
#         draw,
#         clust,
#         10,
#         h / 2,
#         scaling,
#         labels,
#         )
#     img.save(jpeg, 'JPEG')


# def drawnode(
#     draw,
#     clust,
#     x,
#     y,
#     scaling,
#     labels,
#     ):
#     if clust.id < 0:
#         h1 = getheight(clust.left) * 20
#         h2 = getheight(clust.right) * 20
#         top = y - (h1 + h2) / 2
#         bottom = y + (h1 + h2) / 2
#     # Line length
#         ll = clust.distance * scaling
#     # Vertical line from this cluster to children
#         draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))

#     # Horizontal line to left item
#         draw.line((x, top + h1 / 2, x + ll, top + h1 / 2), fill=(255, 0, 0))

#     # Horizontal line to right item
#         draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0,
#                   0))

#     # Call the function to draw the left and right nodes
#         drawnode(
#             draw,
#             clust.left,
#             x + ll,
#             top + h1 / 2,
#             scaling,
#             labels,
#             )
#         drawnode(
#             draw,
#             clust.right,
#             x + ll,
#             bottom - h2 / 2,
#             scaling,
#             labels,
#             )
#     else:
#     # If this is an endpoint, draw the item label
#         draw.text((x + 5, y - 7), labels[clust.id], (0, 0, 0))


# def rotatematrix(data):
#     newdata = []
#     for i in range(len(data[0])):
#         newrow = [data[j][i] for j in range(len(data))]
#         newdata.append(newrow)
#     return newdata


# def kcluster(rows, distance=pearson, k=4):
#   # Determine the minimum and maximum values for each point
#     ranges = [(min([row[i] for row in rows]), max([row[i] for row in rows]))
#               for i in range(len(rows[0]))]

#   # Create k randomly placed centroids
#     clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
#                 for i in range(len(rows[0]))] for j in range(k)]

#     lastmatches = None
#     for t in range(100):
#         print('Iteration {0}', t)
#         bestmatches = [[] for i in range(k)]

#     # Find which centroid is the closest for each row
#         for j in range(len(rows)):
#             row = rows[j]
#             bestmatch = 0
#             for i in range(k):
#                 d = distance(clusters[i], row)
#                 if d < distance(clusters[bestmatch], row):
#                     bestmatch = i
#             bestmatches[bestmatch].append(j)

#     # If the results are the same as last time, this is complete
#         if bestmatches == lastmatches:
#             break
#         lastmatches = bestmatches

#     # Move the centroids to the average of their members
#         for i in range(k):
#             avgs = [0.0] * len(rows[0])
#             if len(bestmatches[i]) > 0:
#                 for rowid in bestmatches[i]:
#                     for m in range(len(rows[rowid])):
#                         avgs[m] += rows[rowid][m]
#                 for j in range(len(avgs)):
#                     avgs[j] /= len(bestmatches[i])
#                 clusters[i] = avgs

#     return bestmatches


# def tanamoto(v1, v2):
#     (c1, c2, shr) = (0, 0, 0)

#     for i in range(len(v1)):
#         if v1[i] != 0:  # in v1
#             c1 += 1
#         if v2[i] != 0:  # in v2
#             c2 += 1
#         if v1[i] != 0 and v2[i] != 0:  # in both
#             shr += 1

#     return 1.0 - float(shr) / (c1 + c2 - shr)

########################################################################
## mnist

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train_images = mnist.train.images
# train_labels = mnist.train.labels

# def is_outlier(points, threshold=3.5):
#     """
#     Returns a boolean array with True if points are outliers and False
#     otherwise.
    
#     Data points with a modified z-score greater than this
#     # value will be classified as outliers.
#     """
#     # transform into vector
#     if len(points.shape) == 1:
#         points = points[:,None]

#     # compute median value    
#     median = np.median(points, axis=0)
    
#     # compute diff sums along the axis
#     diff = np.sum((points - median)**2, axis=-1)
#     diff = np.sqrt(diff)
#     # compute MAD
#     med_abs_deviation = np.median(diff)
    
#     # compute modified Z-score
#     # http://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Iglewicz
#     modified_z_score = 0.6745 * diff / med_abs_deviation

#     # return a mask for each outlier
#     return modified_z_score > threshold

from pooh import utils as ut
# res = ut.http_post(url="http://m2.shilladfs.com/osd/kr/zh",
#                     postParams={'lang':'zh', 'applang':'62', 'ostype':'I', 'modelname':'iPhone8,2',
#                                 'deviceuuid':'93976372-57A9-45BD-AAF2-9D91F22B7102',  'autologin':'0',
#                                 'userid':'', 'token':'', 'uuid':'686F7465-6C73-6869-6C6C-6173656FFE6D'})
headers={'http_org':'(null)', 'appLang':'62', 'osType':'I', 'modelName':'iPhone8,2', 'deviceUuid':'93976372-57A9-45BD-AAF2-9D91F22B7102'}
res = ut.http_post(url="http://m2.shilladfs.com/osd/kr/zh/", headers=headers,
                    postParams=r"lang=zh&applang=62&ostype=I&modelname=iPhone8,2&deviceuuid=93976372-57A9-45BD-AAF2-9D91F22B7102&autologin=0&userid=&token=&uuid=686F7465-6C73-6869-6C6C-6173656FFE6D")
f = open('/Users/linked0/akaggle/res.txt', 'w')
f.write(res.text.encode('utf-8'))
f.close()