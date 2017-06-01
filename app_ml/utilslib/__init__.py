import os

def list():
	path = __file__
	path = path[:path.rfind('/')]
	for dirname, dirnames, filenames in os.walk(path):
		# print path to all subdirectories first.
		for subdirname in dirnames:
			print(subdirname)

def list2(folder):
	onceprint = False
	path = __file__
	path = path[:path.rfind('/')]
	for dirname, dirnames, filenames in os.walk(path):
		# print path to all filenames.
		if dirname.find(folder) >=0:
			if onceprint == False:
				print('<< ', dirname, ' >>')
				onceprint = True
			for filename in filenames:
				print(filename)

def loc():
	path = __file__
	return path[:path.rfind('/')]
