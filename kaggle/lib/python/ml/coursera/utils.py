import os

def get_folder(path):
	path = path[:path.rfind('/')]
	return path