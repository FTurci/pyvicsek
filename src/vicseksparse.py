import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def dump(fw, pos,cos, sin, meta=None):
	fw.write("%d\n"%pos.shape[0])
	if meta is not None:
		for key,value in meta.items():
			fw.write(f"{key}:{value}, ")
	else:
		fw.write("Particles\n")

	for i in range(pos.shape[0]):
		fw.write("A %g %g 0 %g %g 0\n"%( pos[i][0], pos[i][1], cos[i], sin[i]))

L = 32.0
ρ = 2.0
N =	int(ρ*L**2)
print(" N",N)

r0 = 1.0
Δt = 1.0
factor =0.5
v0 = r0/Δt*factor
iterations = 10000
η = 0.10


pos = np.random.uniform(0,L,size=(N,2))
orient = np.random.uniform(-np.pi, np.pi,size=N)

fhd = open("tj.xyzo", 'w')

for i in range(iterations):
	print(i)

	tree = cKDTree(pos,boxsize=[L,L])
	dist = tree.sparse_distance_matrix(tree, max_distance=r0,output_type='coo_matrix')

	#important 3 lines: we evaluate a quantity for every column j
	data = np.exp(orient[dist.col]*1j)
	# construct  a new sparse marix with entries in the same places ij of the dist matrix
	neigh = sparse.coo_matrix((data,(dist.row,dist.col)), shape=dist.get_shape())
	# and sum along the columns (sum over j)
	S = np.squeeze(np.asarray(neigh.tocsr().sum(axis=1)))
	
	
	orient = np.angle(S)+η*np.random.uniform(-np.pi, np.pi, size=N)


	cos, sin= np.cos(orient), np.sin(orient)
	pos[:,0] += cos*v0
	pos[:,1] += sin*v0

	pos[pos>L] -= L
	pos[pos<0] += L

	if i%10==0:
		dump(fhd,pos,cos, sin)



