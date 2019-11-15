import math


class PriorityQueue:
	def __init__(self):
		self.values = []
		self.keys = []

	def add(self, key, value):
		if len(self.values) == 0:
			self.values.append(value)
			self.keys.append(key)
		else:
			index = 0
			while index < len(self.values) and key > self.keys[index]:
				index += 1

			if index == len(self.values):
				self.values.append(value)
				self.keys.append(key)
			else:
				self.values.insert(index, value)
				self.keys.insert(index, key)

	def peek(self):
		return self.values[0]

	def poll(self):
		if len(self.values) == 0:
			return None
		else:
			self.keys.pop()
			return self.values.pop(0)


def calculate_distance(a, b):
	if len(a) != len(b):
		raise AttributeError('two tuples have mismatching lengths: %d and %d' % (len(a), len(b)))
	diffs = [0] * len(a)
	for index, (a_i, b_i) in enumerate(zip(a, b)):
		diffs[index] = a_i - b_i
	return math.sqrt(sum([diff_l ** 2 for diff_l in diffs]) / len(diffs))


def gaussian_kernel_function(u, h):
	return 1 / (2.5066282746310002 * h * math.exp(u * u / (2 * h * h)))


def epanechnikov_kernel_function(u, h):
	return 0.75 / h * (1 - u * u / (h * h))


def calculate_mean(values):
	return sum(values) / len(values)


def calculate_z_score(target, values):
	mean = calculate_mean(values=values)
	std_dev = math.sqrt((1 / len(values)) * sum((value - mean) ** 2 for value in values))
	return (target - mean) / std_dev


def sigmoid(x):
	return 1 / (1 + math.exp(-x))
