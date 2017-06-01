import numpy as np
import requests

######################################################################
## utils.py
from itertools import izip
from scipy.sparse import coo_matrix

def intToStr(n, base, alphabet):
    def toStr(n, base, alphabet):
        return alphabet[n] if n < base else toStr(n//base,base,alphabet) + alphabet[n%base]
    return ('-' if n < 0 else '') + toStr(abs(n), base, alphabet)

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def compare_file_data(filepath1, filepath2):
	samedata = True
	data1 = np.loadtxt(filepath1)
	data2 = np.loadtxt(filepath2)
	if data1.shape != data2.shape:
		print ('dimensions of two data are NOT same')
		print ('file1 shape:', data1.shape)
		print ('file2 shape:', data2.shape)
		return

	rcnt = data1.shape[0]
	ccnt = 0
	if len(data1.shape) > 1:
		ccnt = data1.shape[1]

	for i in range(rcnt):
		if ccnt == 0:
			if isclose(data1[i], data2[i]) == False:
				samedata = False
				print('NOT same data row: ', (i))
				print('data1: ', data1[i])
				print('data2: ', data2[i])
		else:
			for j in range(ccnt):
				if isclose(data1[i, j], data2[i,j]) == False:
					samedata = False
					print ('NOT same data row:%d, col:%d', (i, j))
					print ('data1: ', data1[i])
					print ('data2: ', data2[i])
	if samedata == True:
		print ('all data are same')
	else:
		print ('NOT all data is same')

def save_temp_file(*args):
	filepath = args[0]
	tempfile = open(filepath, 'w')
	datalist = list(args[1:])
	print ('Not Yet Implemented!!!')
	tempfile.close()

def sort_sparse_matrix(d):
	m = coo_matrix(d)
  	tuples = izip(m.row, m.col, m.data)
 	return sorted(tuples, key=lambda x: (x[0], x[2]), reverse=True)

######################################################################
##Kevin Murphy utils.py

import os
import scipy.io as sio
import glob

PYTHON_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PYTHON_DIR), 'pmtkdataCopy')


def add_ones(X):
    """Add a column of ones to X"""
    n = len(X)
    return np.column_stack((np.ones(n), X))


def degexpand(X, deg, add_ones=False):
    """Expand input vectors to contain powers of the input features"""
    n = len(X)
    xx = X
    for i in range(1, deg):
        xx = np.column_stack((xx, np.power(X, i + 1)))
    if add_ones:
        xx = np.column_stack((np.ones(n), xx))

    return xx


def rescale_data(X, min_val=-1, max_val=1, minx=None, rangex=None):
    """
    Rescale columns to lie in the range
    [min_val, max_val] (defaults to [-1,1]])
    """
    if minx is None:
        minx = X.min(axis=0)
    if rangex is None:
        rangex = X.max(axis=0) - X.min(axis=0)

    return (max_val - min_val) * (X - minx) / rangex + min_val


def center_cols(X, mu=None):
    """
    Make each column be zero mean
    """
    if mu is None:
        mu = X.mean(axis=0)
    return X - mu, mu


def mk_unit_variance(X, s=None):
    """
    Make each column of X be variance 1
    """
    if s is None:
        s = X.std(axis=0)

    try:
        len(s)
        s[s < np.spacing(1)] = 1
    except TypeError:
        s = s if s > np.spacing(1) else 1

    return X / s, s


class preprocessor_create():
    def __init__(self, standardize_X=False, rescale_X=False, kernel_fn=None,
                 poly=None, add_ones=False):
        self.standardize_X = standardize_X
        self.rescale_X = rescale_X
        self.kernel_fn = kernel_fn
        self.poly = poly
        self.add_ones = add_ones


def poly_data_make(sampling="sparse", deg=3, n=21):
    """
    Create an artificial dataset
    """
    np.random.seed(0)

    if sampling == "irregular":
        xtrain = np.concatenate(
            (np.arange(-1, -0.5, 0.1), np.arange(3, 3.5, 0.1)))
    elif sampling == "sparse":
        xtrain = np.array([-3, -2, 0, 2, 3])
    elif sampling == "dense":
        xtrain = np.arange(-5, 5, 0.6)
    elif sampling == "thibaux":
        xtrain = np.linspace(0, 20, n)
        xtest = np.arange(0, 20, 0.1)
        sigma2 = 4
        w = np.array([-1.5, 1/9.])
        fun = lambda x: w[0]*x + w[1]*np.square(x)

    if sampling != "thibaux":
        assert deg < 4, "bad degree, dude %d" % deg
        xtest = np.arange(-7, 7, 0.1)
        if deg == 2:
            fun = lambda x: (10 + x + np.square(x))
        else:
            fun = lambda x: (10 + x + np.power(x, 3))
        sigma2 = np.square(5)

    ytrain = fun(xtrain) + np.random.normal(0, 1, xtrain.shape) * \
        np.sqrt(sigma2)
    ytestNoisefree = fun(xtest)
    ytestNoisy = ytestNoisefree + np.random.normal(0, 1, xtest.shape) * \
        np.sqrt(sigma2)

    return xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2        


def load_mat(matName):
    """look for the .mat file in pmtk3/pmtkdataCopy/
    currently only support .mat files create by Matlab 5,6,7~7.2,
    """
    try:
        data = sio.loadmat(os.path.join(DATA_DIR, matName))
    except NotImplementedError:
        raise
    except FileNotFoundError:
        raise
    return data


def generate_rst():
    """generate chX.rst in current working directory"""
    cwd = os.getcwd()
    demo_dir = os.path.join(cwd, 'demos')
    chapters = os.listdir(demo_dir)
    for chapter in chapters:
        if not os.path.isdir(os.path.join(demo_dir, chapter)):
            continue
        reg_py = os.path.join(demo_dir, chapter, '*.py')
        scripts = glob.glob(reg_py)
        rst_file = chapter + '.rst'
        rst_file = os.path.join(demo_dir, chapter, rst_file)
        with open(rst_file, 'w') as f:
            f.write(chapter)
            f.write('\n========================================\n')
            for script in scripts:
                script_name = os.path.basename(script)
                f.write('\n' + script_name[:-3])
                f.write('\n----------------------------------------\n')
                reg_png = os.path.join(demo_dir,
                                       chapter,
                                       script_name[:-3] + '*.png')
                for img in glob.glob(reg_png):
                    img_name = os.path.basename(img)
                    f.write(".. image:: " + img_name + "\n")
                f.write(".. literalinclude:: " + script_name + "\n")

######################################################################
## stats.py
def findna(org_data):
	find_null(org_data)

def find_null(org_data):
  data = org_data.isnull().values
  for j in range(data.shape[1]):
    li = list()
    print('{0} -> Indeces of NaN'.format(org_data.columns[j]))
    for i in range(data.shape[0]):
      if data[i, j] == True:
        li.append(i)
    if len(li) > 0:
    	print('\t{0}'.format(li))
    else:
    	print('\tNo NaNs')

def is_outlier(points, threshold=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.
    
    Data points with a modified z-score greater than this
    # value will be classified as outliers.
    """
    # transform into vector
    if len(points.shape) == 1:
        points = points[:,None]

    # compute median value    
    median = np.median(points, axis=0)
    
    # compute diff sums along the axis
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    # compute MAD
    med_abs_deviation = np.median(diff)
    
    # compute modified Z-score
    # http://www.itl.nist.gov/div898/handbook/eda/section4/eda43.htm#Iglewicz
    modified_z_score = 0.6745 * diff / med_abs_deviation

    # return a mask for each outlier
    return modified_z_score > threshold
	
######################################################################
## google_search.py
def googlesearch(searchfor):
    link = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&%s' % searchfor    
    ua = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.116 Safari/537.36'}
    payload = {'q': searchfor}                                                 
    response = requests.get(link, headers=ua, params=payload)
    return response.text 

######################################################################
## file.py
import os
excl_list = ['.DS_Store'];

def get_folder(path):
	path = path[:path.rfind('/')]
	return path

def get_files(path):
	file_list = []
	for dirname, dirnames, filenames in os.walk(path):
		#filenames = filenames - excl_list
		file_list += filenames
	return file_list

def get_packages():
    file_list = []
    for dirname, dirnames, filenames in os.walk("/Users/linked0/libpy"):
        #filenames = filenames - excl_list
		file_list += [dirname+'/'+filename for filename in filenames if filename.endswith(".py")]
    return file_list

######################################################################
## map.py

from geopy.distance import vincenty
import geocoder

def distance(start_lat, start_lon, end_lat, end_lon):
    start = (start_lat, start_lon)
    end = (end_lat, end_lon)
    dist = vincenty(start, end).meters
    print "dist: %f" % (dist)

def get_latlng(addr):
	g = geocoder.google(addr)
	return g.latlng

def milesdistance(a1,a2):
 	lat1,long1=getlocation(a1)
  	lat2,long2=getlocation(a2)
  	latdif=69.1*(lat2-lat1)
	longdif=53.0*(long2-long1)
	return (latdif**2+longdif**2)**.5

######################################################################
# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import numpy as np

def load_dataset(dataset_name):
    '''
    data,labels = load_dataset(dataset_name)

    Load a given dataset

    Returns
    -------
    data : numpy ndarray
    labels : list of str
    '''
    data = []
    labels = []
    fld = get_folder(__file__)
    with open(fld + '/data/{0}.tsv'.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# This function was called ``learn_model`` in the first edition
def fit_model(features, labels):
    '''Learn a simple threshold model'''
    best_acc = -1.0
    # Loop over all the features:
    for fi in range(features.shape[1]):
        thresh = features[:, fi].copy()
        # test all feature values in order:
        thresh.sort()
        for t in thresh:
            pred = (features[:, fi] > t)

            # Measure the accuracy of this 
            acc = (pred == labels).mean()

            rev_acc = (pred == ~labels).mean()
            if rev_acc > acc:
                acc = rev_acc
                reverse = True
            else:
                reverse = False
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse

    # A model is a threshold and an index
    print('best_t: {0}, best_fi: {1}'.format(best_t, best_fi))
    return best_t, best_fi, best_reverse


# This function was called ``apply_model`` in the first edition
def predict(model, features):
    '''Apply a learned model'''
    # A model is a pair as returned by fit_model
    t, fi, reverse = model
    if reverse:
        return features[:, fi] <= t
    else:
        return features[:, fi] > t

def accuracy(features, labels, model):
    '''Compute the accuracy of the model'''
    preds = predict(model, features)
    return np.mean(preds == labels)

######################################################################next
import requests
def http_post(url, headers=None, postParams=None):
    """
    headers = {'user-agent': 'my-app/0.0.1'}
    res = requests.get(url, headers=headers)
    r = requests.post('http://httpbin.org/post', data = {'key':'value'})
    r = requests.put('http://httpbin.org/put', data = {'key':'value'})
    r = requests.delete('http://httpbin.org/delete')
    r = requests.head('http://httpbin.org/get')
    r = requests.options('http://httpbin.org/get')

    payload = {'key1': 'value1', 'key2': 'value2'}
    r = requests.get('http://httpbin.org/get', params=payload)
    """

    res = requests.post(url, headers=headers, data=postParams.encode('utf-8'))
    return res

def test(data):
    print("test: ", data)

######################################################################
## main
if __name__ == '__main__':
    generate_rst()
    print("Finished generate chX.rst!")

