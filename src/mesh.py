# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 19:56:24 2023

@author: Matthieu Nougaret
"""
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

def id_start_end_nods(n_cycle):
	"""
	Find what are the indices of the starting (higger left) nod and the ending
	(lower right) nod.

	Parameters
	----------
	n_cycle : int
		Number of embending cycle put to create the mesh. It is the 'agrand'
		parameter in HexaMesh function.

	Returns
	-------
	id_s : int
		Indice of the starting (higger left) nod.
	id_e : int
		Indice of the ending (lower right) nod.

	Example
	-------
	In[0] : id_start_end_nods(3)
	Out[0] : (25, 34)

	"""
	id_s = 3*n_cycle**2 - n_cycle + 1
	id_e = 3*n_cycle**2 + 2*n_cycle + 1
	return id_s, id_e

def create_mesh(n, method):
	"""
	Function to create a triangular mesh with a shape that is roughly squared. 

	Parameters
	----------
	n : int
		Number of layer that we want to have. Consequently, the output number
		of dots will be n**2.
	method : str
		Method use for the creation of the mesh.

	Returns
	-------
	positions : numpy.ndarray
		Positions of the dots. Shape is (n, 2) with n the number of dots. The
		first column is for the x-axis positions and the second for the y-axis
		positions.

	Example
	-------
	In[0] : create_mesh(5, 'square')
	Out[0] : array([[0. , 0.        ], [1. , 0.        ], [2. , 0.        ],
					[3. , 0.        ], [4. , 0.        ], [0.5, 0.8660254 ],
					[1.5, 0.8660254 ], [2.5, 0.8660254 ], [3.5, 0.8660254 ],
					[4.5, 0.8660254 ], [0. , 1.73205081], [1. , 1.73205081],
					[2. , 1.73205081], [3. , 1.73205081], [4. , 1.73205081],
					[0.5, 2.59807621], [1.5, 2.59807621], [2.5, 2.59807621],
					[3.5, 2.59807621], [4.5, 2.59807621], [0. , 3.46410162],
					[1. , 3.46410162], [2. , 3.46410162], [3. , 3.46410162],
					[4. , 3.46410162]])

	"""
	if method == 'square':
		x = np.zeros((n, n))
		x[np.arange(n)%2 == 0] = np.arange(n)
		x[np.arange(n)%2 == 1] = np.arange(0.5, n)
		y = np.ones((n, n))*np.arange(n)[:, np.newaxis]*3**.5 /2
		x = np.ravel(x)
		y = np.ravel(y)

	elif method == 'circle':
		# Generate the branches, corresponding to angles from 0 to 360 at each 60°
		ray = 1
		pns = [np.array([[0, 0]])]
		for i in range(n):
			# radian angles
			theta = np.arange(0, 360, 60)/360*2*np.pi
			# dots
			Npns = np.zeros((len(theta), 2))
			Npns[:, 0] = np.cos(theta)*ray
			Npns[:, 1] = np.sin(theta)*ray
			pns.append(Npns)
			ray += 1
	
		# Interpollation by adding dots between the branches
		for i in range(2, n+1):
			llp = np.zeros((int(6*i+1), 2))
			di = np.arange(int(6*i+1))
			ins = di[di%i == 0][:-1]
			sto = di[di%i == 0]
			llp[ins, 0] = pns[i][:, 0]
			llp[ins, 1] = pns[i][:, 1]
			llp[-1] = pns[i][0]
	
			for j in range(len(sto)-1):
				p0 = llp[sto[j+1]]
				p1 = llp[sto[j]]
				dx = p1[0]-p0[0]
				dy = p1[1]-p0[1]
				a = (p1[1]-p0[1])/dx
				b = (p0[1]*p1[0]-p1[1]*p0[0])/dx
				# Rounding of calculated positions 
				a = float(format(a, '.12f'))
				b = float(format(b, '.12f'))
				ptp = (np.array([np.arange(i-1), np.arange(i-1)]).T+1)/i
				if j <= (len(sto)-2)/2:
					ptp[:, 0] = ptp[:, 0]*dx+np.min([p0[0], p1[0]])
				else:
					ptp[:, 0] = ptp[:, 0]*dx+np.max([p0[0], p1[0]])
	
				ptp[:, 1] = ptp[:, 0]*a+b
				out = np.arange(sto[j]+1, sto[j+1])
				llp[out] = ptp
	
			pns[i] = llp[:-1]
	
		alli = np.concatenate(np.array(pns, dtype=object))
		x, y = alli.T

	positions = np.array([x, y]).T
	return positions

def groupe_by(positions):
	"""
	Function to pass from a vectorised dictionary of pairwise links between
	dots, to an other vectorised dictionary of full connection for each dots.

	Parameters
	----------
	positions : numpy.ndarray
		Positions of the dots. Shape is (n, 2) with n the number of dots. The
		first column is for the x-axis positions and the second for the y-axis
		positions.

	Returns
	-------
	vetcorised_dico : numpy.ndarray
		A 2-dimensional array that store all the indices of the connected dots
		for each of them.

	Example
	-------
	In[0] : positions = create_mesh(5, 'square')
	In[1] : groupe_by(positions)
	Out[1] : array([array([0, 1, 5], dtype=int64),
					array([1, 2, 5, 6, 0], dtype=int64),
					array([2, 3, 6, 7, 1], dtype=int64),
					array([3, 4, 7, 8, 2], dtype=int64),
					array([4, 8, 9, 3], dtype=int64),
					array([ 5,  6, 10, 11,  0,  1], dtype=int64),
					array([ 6,  7, 11, 12,  1,  2,  5], dtype=int64),
					array([ 7,  8, 12, 13,  2,  3,  6], dtype=int64),
					array([ 8,  9, 13, 14,  3,  4,  7], dtype=int64),
					array([ 9, 14,  4,  8], dtype=int64),
					array([10, 11, 15,  5], dtype=int64),
					array([11, 12, 15, 16,  5,  6, 10], dtype=int64),
					array([12, 13, 16, 17,  6,  7, 11], dtype=int64),
					array([13, 14, 17, 18,  7,  8, 12], dtype=int64),
					array([14, 18, 19,  8,  9, 13], dtype=int64),
					array([15, 16, 20, 21, 10, 11], dtype=int64),
					array([16, 17, 21, 22, 11, 12, 15], dtype=int64),
					array([17, 18, 22, 23, 12, 13, 16], dtype=int64),
					array([18, 19, 23, 24, 13, 14, 17], dtype=int64),
					array([19, 24, 14, 18], dtype=int64),
					array([20, 21, 15], dtype=int64),
					array([21, 22, 15, 16, 20], dtype=int64),
					array([22, 23, 16, 17, 21], dtype=int64),
					array([23, 24, 17, 18, 22], dtype=int64),
					array([24, 18, 19, 23], dtype=int64)], dtype=object)

	"""
	kern = np.arange(positions.shape[0], dtype=int)
	vetcorised_dico = list(kern[:, np.newaxis])
	distances = cdist(positions, positions).astype('float32')
	for i in range(positions.shape[0]):
		vetcorised_dico[i] = np.append(vetcorised_dico[i],
										kern[distances[i] == 1])

	vetcorised_dico = np.array(vetcorised_dico, dtype=object)
	return vetcorised_dico

def make_first_conn(vect_dico):
	"""
	Function to make the first connection between dots.

	Parameters
	----------
	vect_dico : numpy.ndarray
		A 2-dimensional array that store all the indices of the connected dots
		for each of them.

	Returns
	-------
	vect_dico_maze : numpy.ndarray
		A 2-dimensional array that store the indices of the connected dots
		for the maze.
	connect_v : numpy.ndarray
		A 1-dimensional array that store the value of the structures to keep
		the evolution of the and wich dots are connected.

	Example
	-------
	In[0] : positions = create_mesh(5, 'square')
	In[1] : adic = groupe_by(positions)
	In[2] : make_first_conn(adic)
	Out[3] : ([array([0, 5]), array([1, 6]), array([2, 3]), array([3, 7, 2]),
			   array([4, 8]), array([5, 0]), array([6, 1]),
			   array([ 7,  3, 12]), array([8, 4]), array([ 9, 14]),
			   array([10, 11, 15]), array([11, 10]), array([12,  7]),
			   array([13, 17]), array([14,  9, 18]), array([15, 10]),
			   array([16, 22]), array([17, 13, 23]), array([18, 14, 24, 19]),
			   array([19, 18]), array([20, 21]), array([21, 20]),
			   array([22, 16]), array([23, 17]), array([24, 18])],
			   array([0, 1, 2, 2, 4, 0, 1, 2, 4, 9, 10, 10, 2, 13, 9, 10, 16,
					 13, 9, 9, 20, 20, 16, 13, 9]))

	"""
	length = len(vect_dico)
	vect_dico_maze = list(np.arange(length, dtype=int)[:, np.newaxis])
	connect_v = np.arange(length)
	drawed = np.arange(length) # To not redraw the same value
	stop = False
	while stop != True:
		r = drawed[np.random.randint(len(drawed))]
		drawed = drawed[drawed != r] # remove the drawed dot
		s = connect_v[r]
		if len(connect_v[connect_v == s]) == 1:
			l = vect_dico[r]
			c = np.random.randint(1, len(l))
			vect_dico_maze[r] = np.append(vect_dico_maze[r], l[c])
			vect_dico_maze[l[c]] = np.append(vect_dico_maze[l[c]],
											 vect_dico_maze[r][0])

			mini = np.min([s, connect_v[l[c]]])
			connect_v[connect_v == connect_v[r]] = mini
			connect_v[connect_v == connect_v[l[c]]] = mini
			# remove the drawed connection
			drawed = drawed[drawed != l[c]]

		else:
			u = np.unique(connect_v, return_counts=True)
			if np.min(u[1]) > 1:
				stop = True

		if len(drawed) == 0:
			stop = True

	return vect_dico_maze, connect_v

def maze_fusion(vect_dico):
	"""
	Function to construct a maze with an triangular mesh and fusion method.

	Parameters
	----------
	vect_dico : numpy.ndarray
		A 2-dimensional array that store all the indices of the connected dots
		for each of them.
	
	Returns
	-------
	v_dico_m : numpy.ndarray
		A 2-dimensional array that store the indices of the connected dots
		for the maze.

	Example
	-------
	In[0] : positions = create_mesh(5, 'square')
	In[1] : adic = groupe_by(positions)
	In[2] : maze_fusion(adic)
	Out[2] : array([array([0, 1], dtype=object), 
					array([1, 2, 5, 0], dtype=object),
					array([2, 1, 3], dtype=object),
					array([3, 2, 4], dtype=object),
					array([4, 8, 3], dtype=object),
					array([5, 1, 10], dtype=object),
					array([6, 12, 7], dtype=object),
					array([7, 6], dtype=object),
					array([8, 13, 9, 4, 14], dtype=object),
					array([9, 8], dtype=object),
					array([10, 15, 5], dtype=object),
					array([11, 16], dtype=object),
					array([12, 6, 13], dtype=object),
					array([13, 8, 12], dtype=object),
					array([14, 18, 19, 8], dtype=object),
					array([15, 10, 16, 20], dtype=object),
					array([16, 11, 15], dtype=object),
					array([17, 23], dtype=object),
					array([18, 14], dtype=object),
					array([19, 24, 14], dtype=object),
					array([20, 21, 15], dtype=object),
					array([21, 20, 22], dtype=object),
					array([22, 21], dtype=object),
					array([23, 17, 24], dtype=object),
					array([24, 23, 19], dtype=object)], dtype=object)

	"""
	v_dico_m, connect_v = make_first_conn(vect_dico)
	stop = False
	while stop != True:
		#get representatives value and size of the structures (connected dots)
		uniq = np.unique(connect_v, return_counts=True)
		if len(uniq[1]) == 1:
			stop = True

		else:
			try:
				# draw one of the smallest structures
				r = np.random.choice(np.where(uniq[1] == np.min(uniq[1]))[0])
				# draw an indice of the dots from the structures
				r2 = np.random.choice(np.where(connect_v == uniq[0][r])[0])
				targ = v_dico_m[r2]
				thco = vect_dico[r2]
				mth = thco[connect_v[thco] != connect_v[uniq[0][r]]]
				v = connect_v[mth]
				numb = np.zeros(len(v))
				for i in range(len(v)):
					numb[i] = uniq[1][uniq[0] == v[i]]

				agm = np.argmin(numb)
				v_dico_m[mth[agm]] = np.append(v_dico_m[mth[agm]], targ[0])
				v_dico_m[targ[0]] = np.append(v_dico_m[targ[0]], mth[agm])
				mini = np.min([connect_v[mth[agm]], connect_v[targ[0]]])
				connect_v[connect_v == connect_v[mth[agm]]] = mini
				connect_v[connect_v == connect_v[targ[0]]] = mini

			except:
				pass

	v_dico_m = np.array(v_dico_m, dtype=object)
	return v_dico_m

def maze_exploration(all_conn_dict, start_node=0):
	"""
	Function to create a maze through the exploration method.

	Parameters
	----------
	all_conn_dict : numpy.ndarray
		A 2-dimensionals array that store the indices of dots with a distances
		between them equak to 1.
	start_node : int
		Indice of the starting node for the exploration.

	Returns
	-------
	vect_dico_maze : numpy.ndarray
		A 2-dimensional array that store the indices of the connected dots
		for the maze.

	Example
	-------
	In[0] : positions = create_mesh(5, 'square')
	In[1] : adic = groupe_by(positions)
	In[2] : maze_exploration(adic, start_node=0)
	Out[2] : array([array([0, 1]), array([1, 0, 5]), array([2, 6, 7]),
					array([3, 7, 4]), array([4, 3, 9]), array([5, 1, 10]),
					array([6, 12, 11, 2]), array([7, 2, 3]),
					array([8, 9, 14]), array([9, 4, 8]), array([10, 5, 15]),
					array([11, 6]), array([12, 17, 6]), array([13, 14, 18]),
					array([14, 8, 13]), array([15, 10, 21]),
					array([16, 22, 17]), array([17, 16, 12]),
					array([18, 13, 23]), array([19, 24]), array([20, 21]),
					array([21, 15, 22, 20]), array([22, 21, 16]),
					array([23, 18, 24]), array([24, 23, 19])], dtype=object)

	Note
	----
	You can put any int value for start_node parameter if it correspond to a
	node indice. It will in all case create a complete maze.
	This maze creation method is faster than the one used in maze_fusion.

	"""
	vect_dico_maze = list(np.arange(len(all_conn_dict), dtype=int)[:, np.newaxis])
	val_connect = np.arange(len(all_conn_dict))
	cordon = np.array([start_node], dtype=int) # backwalking
	node = start_node
	explore = True
	backwalk = False
	stop = False
	while stop != True:
		if len(val_connect[val_connect != val_connect[start_node]]) == 0:
			explore = False
			backwalk = False
			stop = True

		while explore:
			sv = val_connect[node]
			current = all_conn_dict[node]
			autour = current[val_connect[current] != sv]
			if len(autour) > 0:
				r = np.random.randint(len(autour))
				vect_dico_maze[node] = np.append(vect_dico_maze[node], autour[r])
				vect_dico_maze[autour[r]] = np.append(vect_dico_maze[autour[r]],
													vect_dico_maze[node][0])
				node = autour[r]
				val_connect[node] = sv
				cordon = np.append(cordon, node)

			else:
				explore = False
				backwalk = True

		if backwalk:
			for i in range(1, len(cordon)+1):
				current = all_conn_dict[cordon[-i]]
				autour = current[val_connect[current] != sv]
				if len(autour) > 0:
					break

			cordon = cordon[:(-i+1)]
			node = cordon[-1]
			explore = True
			backwalk = False

	vect_dico_maze = np.array(vect_dico_maze, dtype=object)
	return vect_dico_maze

def kurskal(points):
	"""
	Function to compute Kruskal's algorithm.

	Parameters
	----------
	node_p : numpy.ndarray
		Position of the nodes. It will be used to compute the connection's
		weight trhough euclidian distance.

	Returns
	-------
	tree : numpy.ndarray
		List of the nodes interconnections. The structure is as follow:
		[self indices nodes from nodes_p, ...list of node connected...].

	Example
	-------
	In [0]: dots = np.random.uniform(-3, 10, (11, 2))
	In [1]: Kruskal_algorithm(dots)
	Out[1]: array([array([0, 3, 7, 6]), array([1, 3]), array([ 2,  8, 10]),
				   array([3, 0, 1]), array([ 4,  6,  9, 10]), array([5, 7]),
				   array([6, 4, 0]), array([7, 5, 0]), array([8, 2]),
				   array([9, 4]), array([10,  2,  4])], dtype=object)

	"""
	# calculates the distance matrix
	m_dist = cdist(points, points, metric='euclidean').T
	length = len(points)
	# list of array
	tree = list(np.arange(length)[:, np.newaxis])
	mask = (np.arange(length)-np.arange(length)[:, np.newaxis]) > 0
	# lists of index matrices
	indices = list(np.meshgrid(range(length), range(length)))
	# vector 1d to track connections in the tree and avoid loop formation
	state = np.arange(length)
	# We flatten the 2d matrix by keeping less than half of the distance
	# matrix not to re-evaluate relationships between pairs of points.
	sort_d = m_dist[mask]
	# The same is done for index matrices
	p_j = indices[0][mask]
	p_i = indices[1][mask]
	# Indices sorted in ascending order by distance values
	rank = np.argsort(sort_d)
	# Sorting indices and distance values
	p_i = p_i[rank]
	p_j = p_j[rank]
	sort_d = sort_d[rank]
	for i in range(len(sort_d)):
		# To have no recontection with loops in the tree
		if state[p_i[i]] != state[p_j[i]]:
			tree[p_i[i]] = np.append(tree[p_i[i]], p_j[i])
			tree[p_j[i]] = np.append(tree[p_j[i]], p_i[i])
			# Update of the 'state' vector
			minima = np.min([state[p_i[i]], state[p_j[i]]])
			state[state == state[p_i[i]]] = minima
			state[state == state[p_j[i]]] = minima
			# early stoping to avoid useless loop
			if len(state[state != minima]) == 0:
				break

	tree = np.array(tree, dtype=object)
	return tree

def kurskal_maze(posisition):
	"""
	Function to create a maze through the computation of a minimum spanning
	tree with Kruskal's algorithm.

	Parameters
	----------
	posisition : numpy.ndarray
		Positions of the dots. Shape is (n, 2) with n the number of dots. The
		first column is for the x-axis positions and the second for the y-axis
		positions.

	Returns
	-------
	kurskal_tree : numpy.ndarray
		A 2-dimensional array that store the indices of the connected dots
		for the maze.

	"""
	weights = posisition+np.random.uniform(-0.3, 0.3, posisition.shape)
	kurskal_tree = kurskal(weights)
	return kurskal_tree

def complexification(all_connections, maze_connections):
	"""
	Function to make random conection between nodes. This will lead to the
	creation of multiple possible path.

	Parameters
	----------
	all_connections : numpy.ndarray
		A 2-dimensionals array that store the indices of dots with a distances
		between them equak to 1.
	maze_connections : numpy.ndarray
		A 2-dimensional array that store the indices of the connected dots
		for the maze.

	Returns
	-------
	maze_connections : numpy.ndarray
		Updated 2-dimensional array that store the indices of the connected
		dots for the maze.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_exploration(adic, start_node=0)
	In[3] : complexification(adic, vdm)
	Out[3] : array([array([0, 5, 1]), array([1, 2, 0]), array([2, 3, 1, 6]),
					array([3, 4, 2]), array([4, 9, 3]),
					array([5, 0, 10, 6]), array([6, 11, 7, 5, 2]),
					array([7, 6, 8]), array([8, 7, 13]), array([9, 14, 4]),
					array([10, 5, 15]), array([11, 12, 6]),
					array([12, 16, 11, 13]), array([13, 8, 14, 12]),
					array([14, 13, 9, 19]), array([15, 10, 21]),
					array([16, 22, 12]), array([17, 18, 22]),
					array([18, 23, 17]), array([19, 14, 24]), array([20, 21]),
					array([21, 15, 22, 20]), array([22, 21, 16, 17]),
					array([23, 24, 18]), array([24, 19, 23])],
				dtype=object)

	"""
	length = len(all_connections)
	for k in range(int(length**.5)):
		rand = np.random.randint(length)
		grad = Dijkstra_triangular_mesh(maze_connections, rand)
		differ = np.zeros((length, 2), dtype=int)
		for i in range(length):
			dh = grad[i]-grad[all_connections[i]]
			differ[i] = np.max(dh), np.argmax(dh)

		maxima = differ[:, 0] == np.max(differ[:, 0])
		posi_max = differ[maxima, 1][0]
		from_ = maze_connections[maxima][0][0]
		to_ = all_connections[from_][posi_max]
		maze_connections[from_] = np.append(maze_connections[from_], to_)
		maze_connections[to_] = np.append(maze_connections[to_], from_)

	return maze_connections

def lin_smooth(x, y, height, kernel_size):
	"""
	Function to make linear smoothing.

	Parameters
	----------
	x : numpy.ndarray
		X-axis positions of the dots.
	y : numpy.ndarray
		Y-axis positions of the dots.
	height : numpy.ndarray
		Z-axis value of the dots.
	kernel_size : int
		Range in which the kernel will be applied.

	Returns
	-------
	smoothed : TYPE
		The smoothed z-axis value of the dots.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : z = np.random.normal(0, 1, len(x))
	In[2] : zp = lin_smooth(x, y, z, 2)
	In[3] : z-zp
	Out[3] : array([ 0.67545505,  0.55178669,  0.71163035, -1.20233883,
					-0.9080009 , -0.99897729, -0.11821925,  1.48719037,
					-1.33534994,  1.30334781, -1.00158817,  0.87744239,
					-0.08104341,  0.11721114, -0.22719919, -1.0836518 ,
					-0.56028265,  0.88224834,  0.34529368, -0.02208778,
					 0.25436259, -0.14806951,  0.68129124, -1.10411634,
					 0.15636067])

	"""
	smoothed = np.zeros(len(height))
	for i in range(len(x)):#tqdm(, desc='smoothing'):
		dist = (((x[i]-x)**2 +(y[i]-y)**2)**0.5).astype('float32')
		# here a linear smooth is done with an equal weight for all of the
		# values. It should be possible to modify it to have a personal smooth
		# with the wanted weights.
		smoothed[i] = np.mean(height[dist <= kernel_size])

	return smoothed

def kernel_smooth(distances, height, kernel_size):
	"""
	Function to make linear smoothing, but the distance used to calculate
	which node to use comes from the node to node distance.

	Parameters
	----------
	distances : numpy.ndarray
		Number of nodes between each of them and the starting node used to
		create the node to node distance in Dijkstra_triangular_mesh.
	height : numpy.ndarray
		Z-axis value of the dots.
	kernel_size : int
		Range in which the kernel will be applied.

	Returns
	-------
	smoothed : numpy.ndarray
		The smoothed z-axis value of the dots.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_fusion(adic)
	In[3] : z = Dijkstra_triangular_mesh(vdm, 0)
	In[4] : zp = kernel_smooth(z, z, 1)
	In[5] : z-zp
	Out[5] : array([-0.66666667, -0.42857143, -0.27272727, -0.1875    ,
					 0.0625    , -0.42857143, -0.27272727, -0.1875    , 
					 0.0625    ,  0.38461538, -0.27272727, -0.27272727,
					-0.1875    ,  0.0625    ,  0.38461538, -0.1875    ,
					-0.1875    ,  0.0625    ,  0.38461538,  0.66666667,
					 0.0625    ,  0.0625    ,  0.0625    ,  0.38461538,
					 0.66666667])

	"""
	smoothed = np.zeros(len(height))
	for i in range(len(height)):
		dist = np.abs(distances-distances[i])
		smoothed[i] = np.mean(height[dist <= kernel_size])
		
	return smoothed

def Dijkstra_triangular_mesh(connections, id_st_node, id_ed_node=None,
							 fill_early=False):
	"""
	Function to compute the Dijkstra algorithm in a triangular mesh.

	Parameters
	----------
	connections : numpy.ndarray
		A 2-dimensions array that store the interconnections between the dots.
	id_st_node : int
		Indice of the starting node for the exploration.
	id_ed_node : int, optional
		Indice of the targeted node for the exploration. When it is reach, the
		exploration will stop. The default is None.
	fill_early : bool, optional
		If the unexplored dots are filled by max+1 value. This can be
		interesting when it explore a huge mesh. The default is False.

	Returns
	-------
	distances : numpy.ndarray
		A 1-dimensional array storing the distances in nodes-to-nodes, for all
		nodes of the maze.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_fusion(adic)
	In[3] : Dijkstra_triangular_mesh(vdm, 0)
	Out[3] : array([0., 1., 2., 3., 4., 1., 2., 3., 4., 5., 2., 2., 3., 4.,
					5., 3., 3., 4., 5., 6., 4., 4., 4., 5., 6.])

	"""
	earl_stop = False
	if type(id_ed_node) == int:
		earl_stop = True

	distances = np.zeros(len(connections))-1
	sub_connects = connections[id_st_node]
	distances[sub_connects] = distances[sub_connects]+1
	sub_connects = connections[sub_connects[1:]]
	stop = False
	compteur = 2
	while stop != True:
		step = np.array([], dtype=int)
		for i in range(len(sub_connects)):
			step = np.concatenate((step,
					   sub_connects[i][distances[sub_connects[i]] == -1]))

		distances[step] = distances[step]+compteur
		sub_connects = connections[step]
		compteur += 1

		if len(distances[distances == -1]) == 0:
			stop = True

		if earl_stop:
			if distances[id_ed_node] != -1:
				stop = True

	distances[distances > -1] = distances[distances > -1]+1
	distances[id_st_node] = 0
	if fill_early:
		distances[distances == -1] = np.max(distances)+1

	return distances

def optimal_path(gradient, connections, id_st_node):
	"""
	Function to get the optimal path through the gradient calculation.

	Parameters
	----------
	gradient : numpy.ndarray
		A 1-dimensional array that store the distance (in nodes number) for
		all the nodes from the node  at index ID_st_node.
	connections : numpy.ndarray
		A 2-dimensions array that store the interconnections between the dots.
	id_st_node : int
		Indice of the starting node for the gradient descent. Consequently the
		path will start from this node to the node with the lowest gradient
		value.

	Returns
	-------
	path_id : numpy.ndarray
		List of the node's indicies to take to solve the maze.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_fusion(adic)
	In[3] : grad = Dijkstra_triangular_mesh(vdm, 24)
	In[4] : optimal_path(grad, vdm, 0)
	Out[4] : array([ 0,  5, 10, 15, 16, 22, 23, 24])

	"""
	self_value = gradient[id_st_node]
	path_id = np.zeros(int(self_value)+1, dtype=int)
	path_id[0] = id_st_node
	for i in range(int(self_value)):
		autour = connections[path_id[i]]
		poss_vals = gradient[autour]
		next_ = autour[poss_vals == np.min(poss_vals[poss_vals != -1])]
		path_id[i+1] = next_[0]

	return path_id

def update_right_hand(id_node, direction, connections, x, y):
	"""
	Function to update the position and the direction of the exploration of
	the maze whe solving it with the right hand method.

	Parameters
	----------
	id_node : int
		Current position of the explorator in node indice.
	direction : int
		Direction at which the "explorator" is looking. There are six possible
		values : 0, 60, 120, 180, 240 or 300.
	connections : numpy.ndarray
		A 2-dimensions array that store the interconnections between the dots.
	x : numpy.ndarray
		X-axis positions of the dots.
	y : numpy.ndarray
		Y-axis positions of the dots.

	Returns
	-------
	direction : int
		The new direction at which the "explorator" is looking. There are six
		possible values : 0, 60, 120, 180, 240 or 300.
	id_node : int
		The updated current position of the explorator in node indice.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_fusion(adic)
	In[3] : update_right_hand(0, 0, vdm, 5)
	Out[3] : (0, 1)

	"""
	current = connections[id_node]
	dx = np.round(x[current[0]]-x[current], 4)
	dy = np.round(y[current[0]]-y[current], 4)
	if direction == 0:
		if sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]

	elif direction == 60:
		if sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]

	elif direction == 120:
		if sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]

	elif direction == 180:
		if sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]

	elif direction == 240:
		if sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]

	elif direction == 300:
		if sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]

	id_node = id_node[0]
	return id_node, direction

def update_left_hand(id_node, direction, connections, x, y):
	"""
	Function to update the position and the direction of the exploration of
	the maze whe solving it with the left hand method.

	Parameters
	----------
	id_node : int
		Current position of the explorator in node indice.
	direction : int
		Direction at which the "explorator" is looking. There are six possible
		values : 0, 60, 120, 180, 240 or 300.
	connections : numpy.ndarray
		A 2-dimensions array that store the interconnections between the dots.
	n : int
		The input number used for the creation of the mesh in the function
		create_mesh.

	Returns
	-------
	direction : int
		The new direction at which the "explorator" is looking. There are six
		possible values : 0, 60, 120, 180, 240 or 300.
	id_node : int
		The updated current position of the explorator in node indice.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_fusion(adic)
	In[3] : update_left_hand(0, 0, vdm, 5)
	Out[3] : (0, 1)

	"""
	current = connections[id_node]
	dx = np.round(x[current[0]]-x[current], 4)
	dy = np.round(y[current[0]]-y[current], 4)
	if direction == 0:
		if sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]

	elif direction == 60:
		if sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]

	elif direction == 120:
		if sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]

	elif direction == 180:
		if sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]

	elif direction == 240:
		if sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]
		elif sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]

	elif direction == 300:
		if sum((dx == -0.5)&(dy == -0.866)) == 1:
			direction = 60
			id_node = current[(dx == -0.5)&(dy == -0.866)]
		elif sum((dx == -1)&(dy == 0)) == 1:
			direction = 0
			id_node = current[(dx == -1)&(dy == 0)]
		elif sum((dx == -0.5)&(dy == 0.866)) == 1:
			direction = 300
			id_node = current[(dx == -0.5)&(dy == 0.866)]
		elif sum((dx == 0.5)&(dy == 0.866)) == 1:
			direction = 240
			id_node = current[(dx == 0.5)&(dy == 0.866)]
		elif sum((dx == 1)&(dy == 0)) == 1:
			direction = 180
			id_node = current[(dx == 1)&(dy == 0)]
		elif sum((dx == 0.5)&(dy == -0.866)) == 1:
			direction = 120
			id_node = current[(dx == 0.5)&(dy == -0.866)]

	id_node = id_node[0]
	return id_node, direction

def hand_solving(connections, x, y, hand, id_start_node, id_end_node,
				 direction):
	"""
	Function to compute the solution of a maze by using 'right' or 'left' hand
	solution between to given nodes.

	Parameters
	----------
	connections : numpy.ndarray
		A 2-dimensional array that store the indices of the connected dots
		for the maze.
	x : numpy.ndarray
		X-axis positions of the dots.
	y : numpy.ndarray
		Y-axis positions of the dots.
	hand : str
		Hand to use to solve the maze.
	id_start_node : int
		Indice of the starting node for the explorator.
	id_end_node : int
		Indice of the node tageted by the explorator.
	direction : int
		Initial direction at which the "explorator" is looking. There are six
		possible vlues : 0, 60, 120, 180, 240 or 300.

	Returns
	-------
	path_id : numpy.ndarray
		Sorted list of the node's indicies to take to solve the maze.

	Example
	-------
	In[0] : x, y = create_mesh(5, 'square')
	In[1] : adic = groupe_by(x, y)
	In[2] : vdm = maze_fusion(adic)
	In[3] : hand_solving(vdm, 'right', 0, 24, 0)
	Out[3] : array([ 0,  1,  6,  2,  6,  1,  0,  5, 10, 15, 11, 12, 16, 17,
					13,  7, 13, 17, 22, 23, 18, 14,  9,  8,  3,  4,  3,  8, 
					9, 14, 18, 24])

	"""
	n = int((connections[-1][0]+1)**.5)
	id_node = id_start_node
	path_id = []
	path_id.append(id_start_node)
	if hand == 'right':
		while id_node != id_end_node:
			id_node, direction = update_right_hand(id_node, direction,
												   connections, x, y)
			path_id.append(id_node)

	elif hand == 'left':
		while id_node != id_end_node:
			id_node, direction = update_left_hand(id_node, direction,
												   connections, x, y)
			path_id.append(id_node)

	path_id = np.array(path_id)
	return path_id

def comptage(a):
	"""
	Function to calculate the distribution of the number of connection between
	nodes by nodes.

	Parameters
	----------
	a : numpy.ndarray
		A 1-dimensional array storing the number of connection that each nodes
		have.

	Returns
	-------
	bins : numpy.ndarray
		A 1-dimensional array that store the number time that a node have i
		connection(s).

	Example
	-------
	In[0] : x, y = create_mesh(51, 'square')
			adic = groupe_by(x, y)
			vdm = maze_fusion(adic)
			len_fusion = []
			for j in range(len(vdmf)):
				len_fusion.append(len(vdm[j])-1)

			len_fusion = np.array(len_fusion)
			comptage(len_fusion)
	Out[0] : array([ 764., 1176.,  567.,   87.,    7.,    0.])

	"""
	bins = np.zeros(6)
	for i in range(6):
		bins[i] = len(a[a == (i+1)])

	return bins
