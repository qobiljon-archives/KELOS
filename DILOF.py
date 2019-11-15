# from Tools import sigmoid
# from Tools import calculate_distance
# import math
#
# # region Global Variables
# window = 1000  # must be multiple of 4
# min_pts = 5
# skip = False
# half_window = int(window / 2)
# quarter_window = int(window / 4)
# dataset = []
# outliers = []
#
#
# # endregion
#
#
# class LOF:
# 	@staticmethod
# 	def calc_distance_euclidean(_vector1, _vector2):
# 		# !!! all passed vector elements to this method must be float values !!!
#
# 		# validate comparability
# 		if len(_vector1) != len(_vector2):
# 			raise AttributeError("Compared vectors have different number of arguments!")
#
# 		# init differences vector
# 		_per_element_distances = [0] * len(_vector1)
#
# 		# compute (each vector element) difference for RMSE (for euclidean distance)
# 		for index, (_value1, _value2) in enumerate(zip(_vector1, _vector2)):
# 			_per_element_distances[index] = _value1 - _value2
#
# 		# compute RMSE (root mean squared error)
# 		return math.sqrt(sum([_val ** 2 for _val in _per_element_distances]) / len(_per_element_distances))
#
# 	@staticmethod
# 	def calc_k_distance(_k, _vector, _dataset):
# 		# TODO: consider caching for more efficient re-computation
#
# 		_distances = {}
# 		for _vector2 in _dataset:
# 			_distance = LOF.calc_distance_euclidean(_vector1=_vector, _vector2=_vector2)
# 			if _distance in _distances:
# 				_distances[_distance].append(_vector2)
# 			else:
# 				_distances[_distance] = [_vector2]
#
# 		_distances = list(sorted(_distances.items()))
# 		_neighbours = []
#
# 		for _distance in _distances[:_k]:
# 			_neighbours += _distance[1]  # extract each neighbor
#
# 		_k_distance = _distances[_k - 1][0] if len(_distances) >= _k else _distances[-1][0]
#
# 		return _k_distance, _neighbours
#
# 	@staticmethod
# 	def calc_k_reachability_distance(_k, _vector1, _vector2, _dataset):
# 		(_k_distance, _neighbours) = LOF.calc_k_distance(_k, _vector2, _dataset)
# 		return max(_k_distance, LOF.calc_distance_euclidean(_vector1=_vector1, _vector2=_vector2))
#
# 	@staticmethod
# 	def calc_local_reachability_density(_k, _vector, _dataset):
# 		(_k_distance, _neighbours) = LOF.calc_k_distance(_k=_k, _vector=_vector, _dataset=_dataset)
# 		_reachability_distances = [0] * len(_neighbours)
#
# 		for _index, _neighbour in enumerate(_neighbours):
# 			_reachability_distances[_index] = LOF.calc_k_reachability_distance(_k=_k, _vector1=_vector, _vector2=_neighbour, _dataset=_dataset)
#
# 		if sum(_reachability_distances) == 0:
# 			# TODO: vector is identical with its neighbors, consider fixing this case!
# 			# returning 'inf' to note that this vector has an issue
# 			return float("inf")
# 		else:
# 			return len(_neighbours) / sum(_reachability_distances)
#
# 	@staticmethod
# 	def calc_local_outlier_factor(_k, _vector, _dataset):
# 		(_k_distance, _neighbours) = LOF.calc_k_distance(_k=_k, _vector=_vector, _dataset=_dataset)
# 		_vector_lrd = LOF.calc_local_reachability_density(_k=_k, _vector=_vector, _dataset=_dataset)
# 		_lrd_ratios = list([0] * len(_neighbours))
#
# 		for _index, _neighbour in enumerate(_neighbours):
# 			_tmp_dataset_without_neighbor = set(_dataset)
# 			_tmp_dataset_without_neighbor.remove(_neighbour)
# 			_neighbours_lrd = LOF.calc_local_reachability_density(_k=_k, _vector=_neighbour, _dataset=_tmp_dataset_without_neighbor)
# 			_lrd_ratios[_index] = _neighbours_lrd / _vector_lrd
#
# 		return sum(_lrd_ratios) / len(_neighbours)
#
#
# def get_k_distance_decisive(vector, dataset, Y):
# 	D = [0] * half_window
# 	for i in range(half_window):
# 		D[i] = Y[i] * LOF.calc_distance_euclidean(vector, dataset[i])
# 	D.sort()
# 	return D[quarter_window - min_pts]
#
#
# def get_phi(y_n):
# 	if y_n > 1:
# 		return (y_n - 1) ** 2
# 	elif y_n < 0:
# 		y_n ** 2
# 	else:
# 		return 0
#
#
# def get_C(n):
# 	def find_KNN(target, k):
# 		point = [{'point': c, 'distance': 0} for c in dataset]
# 		for i in range(len(point)):
# 			point[i]['distance'] = calculate_distance(a=target, b=point[i]['point'].get_centroid())
# 		point.sort(key=lambda dist_key: dist_key['distance'])
# 		return [point['point'] for point in point[:k]]
#
# 	N = find_KNN()
#
# 	S = [0] * len(dataset)
# 	for n in range(len(dataset)):
# 		for q in N[n]:
# 			S[n] += math.exp(sigmoid(LOF(min_pts, q)))
# 	avg_S = sum(S) / len(dataset)
# 	C_indices, C = [], []
# 	for i in range(len(dataset)):
# 		if S[i] > avg_S:
# 			for n in range(len(dataset)):
# 				if get_k_distance_decisive():
# 					pass
# 	return C_indices, C
#
#
# def get_LOD():
# 	return 0.0
#
#
# def get_NDS(X, step, regularization, iterations):
# 	Y = [0.5] * half_window
# 	for iteration in range(iterations):
# 		step *= 0.95
# 		for n in range(half_window):
# 			C_indices, C = get_C(n)
# 			Y[n] = Y[n] - step * (
# 					sum([Y[i] for i in C_indices]) +
# 					get_k_distance_decisive(X[n], X, Y) / get_k_distance_decisive(X[n], C, Y) -
# 					math.exp(LOF(min_pts, X[n])) +  # TODO: Calculating LOF here doesn't make sense, it's just as bad as using classic LOF
# 					get_phi(Y[n]) +
# 					regularization * (sum([Y[i] for i in range(half_window)]) - quarter_window)
# 			)
# 	Z = []
# 	for n in range(half_window):
# 		if Y[n] == 1:
# 			Z += [X[n]]
# 	return Z
#
#
# def get_DILOF(_vector, threshold, step, regularization, iterations):
# 	global dataset
#
# 	lof_value = get_LOD(_vector, o, outliers, threshold)
# 	print(_vector, lof_value)
#
# 	if lof_value > 0:
# 		dataset += [_vector]
# 		if len(dataset) == window:
# 			Z = get_NDS(dataset, step, regularization, iterations)
# 			dataset = Z + dataset[window / 2:]
#
#
# if __name__ == '__main__':
# 	_vector = (1, 2, 3, 4)
# 	get_DILOF(_vector, 1.0, 1.0, 1.0, 1)
