from KELOS import Cluster
from Tools import PriorityQueue

q = PriorityQueue()

q.add(key=1, value=Cluster(d=(1, 1)))
q.add(key=0, value=Cluster(d=(0, 0)))
q.add(key=-1, value=Cluster(d=(-1, -1)))
q.add(key=-2, value=Cluster(d=(-2, -2)))
q.add(key=2, value=Cluster(d=(2, 2)))
q.add(key=1, value=Cluster(d=(1, 1)))

for key, value in zip(q.keys, q.values):
	print(key, value.get_centroid())

print('peek', q.peek().get_centroid())
print('poll', q.poll().get_centroid())
print('peek', q.peek().get_centroid())
