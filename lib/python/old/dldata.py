import numpy
import theano
import theano.tensor as T

def shared_dataset(data_xy):
	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
	shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

	return shared_x, T.cast(shared_y, 'int32')

