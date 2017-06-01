from math import sqrt
import math
import numpy as np

def list():
    return ('normalizescores(scores, smallIsBetter=0)', 
        'pearson(v1,v2)',
        'euclidean(v1,v2)',
        'gaussian(dist,sigma=10.0)',
        'randrange(n, vmin, vmax)',
        'desc(name)')

def normalizescores(scores, smallIsBetter=0):
    vsmall = 0.00001
    if smallIsBetter:
        minscore = min(scores.values())
        return dict([(u, float(minscore) / max(vsmall, l)) for (u, l) in scores.items()])
    else:
        maxscore = max(scores.values())
        if maxscore == 0: maxscore = vsmall
    return dict([(u, float(c)/maxscore) for (u, c) in scores.items()])

def pearson(v1,v2):
    # Simple sums
    sum1=sum(v1)
    sum2=sum(v2)

    # Sums of the squares
    sum1Sq=sum([pow(v,2) for v in v1])
    sum2Sq=sum([pow(v,2) for v in v2])

    # Sum of the products
    pSum=sum([v1[i]*v2[i] for i in range(len(v1))])

    # Calculate r (Pearson score)
    num=pSum-(sum1*sum2/len(v1))
    den=sqrt((sum1Sq-pow(sum1,2)/len(v1))*(sum2Sq-pow(sum2,2)/len(v1)))
    if den==0: return 0

    return 1.0-num/den

def euclidean(v1, v2):
    d = 0.0
    for i in range(len(v1)):
        d += (v1[i] - v2[i]) ** 2
    return math.sqrt(d)

def jaccard(v1, v2):
    print "not implemented"

def manhattan(v1, v2):
    print "not implemented"

def gaussian(dist,sigma=10.0):
    return math.e**(-dist**2/(2*sigma**2))

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

def desc(name):
    if name == 'pearson':
        return "return distance, smaller the distance, bigger the relavance"
    else:
        return "God only knows"


        