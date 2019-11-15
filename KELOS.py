from Tools import calculate_distance
from Tools import gaussian_kernel_function
from Tools import calculate_z_score
from Tools import PriorityQueue

import math


class Cluster:
	def __init__(self, d):
		self.M = 1  # cardinality of the cluster
		self.LS = []  # linear sum of the points
		self.R_min = []  # minimum values per dimension
		self.R_max = []  # minimum values per dimension
		self.Data = [d]
		for l in range(len(d)):
			self.LS.append(d[l])
			self.R_min.append(d[l])
			self.R_max.append(d[l])

	def get_cardinality(self):
		return self.M

	def insert_point(self, d):
		self.M += 1
		for l in range(len(self.LS)):
			self.LS[l] += d[l]
			self.R_min[l] = min(self.R_min[l], d[l])
			self.R_max[l] = max(self.R_max[l], d[l])
		self.Data += [d]

	def get_centroid(self):
		return [LS_i / self.M for LS_i in self.LS]

	def calc_distance_to(self, d):
		return calculate_distance(d, self.get_centroid())

	def get_radius(self):
		diffs = [0] * len(self.R_max)
		for index, (_max, _min) in enumerate(zip(self.R_max, self.R_min)):
			diffs[index] = _max - _min
		return max(diffs) / 2


def data_abstractor(data, cluster_distance_threshold, clusters):
	# data clusterer function
	print('clustering data of length', len(data))
	for d in data:
		dist_min = float('inf')
		c_n = None
		for c in clusters:
			dist = c.calc_distance_to(d=d)
			if dist < dist_min:
				dist_min = dist
				c_n = c
		if dist_min < cluster_distance_threshold and c_n is not None:
			c_n.insert_point(d)
		else:
			c_new = Cluster(d)
			clusters += [c_new]
	print('data abstraction completed')


def density_estimator(target, clusters, k, cluster_radius=0):
	# weighted kernel density estimator function

	num_of_neighbor_pts = sum([c.get_cardinality() for c in clusters])
	data_dimension = len(target)
	kc_weights = {c: c.get_cardinality() / num_of_neighbor_pts for c in clusters}

	def calc_smoothing_factor(l):
		# finding mean on the l-th layer
		kcs_lth_mean = sum([kc_weights[kc] * kc.get_centroid()[l] for kc in clusters]) / k
		# finding standard deviation on the l-th layer
		kcs_lth_weighted_std_dev = math.sqrt(sum([kc_weights[c] * ((c.get_centroid()[l] - kcs_lth_mean) ** 2) for c in clusters]))
		# calculating smoothing factor
		if kcs_lth_weighted_std_dev == 0:
			kcs_lth_weighted_std_dev = 0.000000000001
		return 1.06 * kcs_lth_weighted_std_dev / (k ** (1 / (data_dimension + 1)))

	local_density: float = 0
	for kc in clusters:
		kernel_products = 1
		kc_centroid = kc.get_centroid()
		for l in range(len(target)):
			kernel_products *= gaussian_kernel_function(abs(target[l] - kc_centroid[l]) + cluster_radius, h=calc_smoothing_factor(l))
		local_density += kc_weights[kc] * kernel_products

	return local_density


def top_n_outlier_detector(clusters, N, k):
	def find_KNN_kernels(x_i):
		_clusters = [{'cluster': c, 'distance': 0} for c in clusters]
		for i in range(len(_clusters)):
			_clusters[i]['distance'] = calculate_distance(a=x_i, b=_clusters[i]['cluster'].get_centroid())
		_clusters.sort(key=lambda dist_key: dist_key['distance'])
		return [kernel['cluster'] for kernel in _clusters[:k]]

	def calc_KLOME(x_i, cluster_radius=0, low=True):
		return calculate_z_score(
			density_estimator(
				target=x_i,
				clusters=find_KNN_kernels(x_i=x_i),
				k=k,
				cluster_radius=cluster_radius if low else -cluster_radius
			),
			[density_estimator(
				target=kc.get_centroid(),
				clusters=find_KNN_kernels(x_i=kc.get_centroid()),
				k=k
			) for kc in clusters]
		)

	print('detecting outliers')

	P = PriorityQueue()
	counter = 0
	for cluster in clusters[:N]:
		counter += 1
		print('pq init', '%.2f' % (counter / N))
		P.add(
			key=calc_KLOME(x_i=cluster.get_centroid(), cluster_radius=cluster.get_radius(), low=False),
			value=cluster
		)

	counter = 0
	num_of_rest_clusters = len(clusters) - N
	for cluster in clusters[N:]:
		print('detecting outliers', '%.2f' % (counter / num_of_rest_clusters))
		counter += 1
		if calc_KLOME(x_i=cluster.get_centroid(), cluster_radius=cluster.get_radius(), low=True) > calc_KLOME(x_i=P.peek().get_centroid(), cluster_radius=P.peek().get_radius(), low=False):
			continue
		elif calc_KLOME(x_i=cluster.get_centroid(), cluster_radius=cluster.get_radius(), low=False) < calc_KLOME(x_i=P.peek().get_centroid(), cluster_radius=P.peek().get_radius(), low=True):
			P.poll()
			P.add(
				key=calc_KLOME(x_i=cluster.get_centroid(), cluster_radius=cluster.get_radius(), low=False),
				value=cluster
			)
		else:
			P.add(
				key=calc_KLOME(x_i=cluster.get_centroid(), cluster_radius=cluster.get_radius(), low=False),
				value=cluster
			)

	print('outliers detected, adding to a list')

	R = PriorityQueue()
	for cluster in P.values:
		for sample in cluster.Data:
			R.add(
				key=calc_KLOME(x_i=sample),
				value=sample
			)
	return R.values


if __name__ == '__main__':
	cluster_distance_threshold = 0.25
	k = 20
	N = 100
	window_size = 1000

	dataset = []
	with open('/Users/kevin/Desktop/KELOS/data.csv', 'r') as r:
		for line in r:
			d0, d1 = line[:-1].split(',')
			dataset += [(float(d0), float(d1))]

	clusters = []
	window_start = 0
	while window_start < len(dataset):
		window = dataset[window_start:window_start + window_size]
		window_start += window_size

		# 1. Data Abstractor => Data Abstractor (including Windows Processor)
		data_abstractor(data=window, cluster_distance_threshold=cluster_distance_threshold, clusters=clusters)
		print('number of clusters:\t', len(clusters))

		# # 2. Density Estimator => Estimator Constructor (including Bandwidth Estimator)
		# local_density = density_estimator(target=target, kernel_centers=kernel_centers, k=3)
		# print("test local density:\t", local_density)

		# 3. Outlier Detector => Top-N Outlier Detector (including Inlier Pruner)
		outliers = top_n_outlier_detector(clusters=clusters, N=N, k=k)
		print(outliers)
