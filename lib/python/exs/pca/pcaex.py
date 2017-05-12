from pooh import file as fl 
from PIL import Image
import numpy
import pylab

fld = fl.get_folder(__file__)
fld = fld + '/gwb_cropped'
imlist = fl.get_files(fld)
print(imlist)

def print_list():
	print(imlist)

def get_list():
	return imlist

def get_fld():
	return fld

im = numpy.array(Image.open(fld + '/' + imlist[0]))
m, n = im.shape[0:2]
imnbr = len(imlist)

immatrix = numpy.array([numpy.array(Image.open(fld + '/' + imlist[i])).flatten() for i in range(imnbr)], 'f')
print(immatrix.shape)

def get_im():
	return im