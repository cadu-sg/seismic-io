from ..Models.InOutSuModel import InOutSu

def readsu(file_path):
	# Read a binary file in .su format
	print("readsu")
	with open(file_path, 'rb') as file:
		return InOutSu.unpack_su(file)