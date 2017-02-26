import pooh

def list():
	path = __file__
	path = path[:path.rfind('/')]
	f = open(path + '/bookex.txt');
	lines = f.readlines()
	for line in lines:
		print(line[:-1])
	f.close()

def loc():
	path = __file__
	return path[:path.rfind('/')]
