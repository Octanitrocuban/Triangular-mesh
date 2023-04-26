# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 20:18:22 2023

@author: Matthieu Nougaret
"""
import numpy as np
import matplotlib.pyplot as plt


def y_limits(a, marge):
	"""
	Function to calculate the lower and upper limits to use in plt.ylim.

	Parameters
	----------
	a : numpy.ndarray
		A n-dimensional numpy.ndarray.
	marge : float
		Must be in the range [0, 1].

	Returns
	-------
	lower : float
		Lower bound.
	upper : float
		Upper bound.

	Exemple
	-------
	In[0] : X, Y = np.arange(10), np.arange(-5, 6)
	In[1] : y_limits(X, 0.05)
	Out[1] : (-0.45, 9.45)
	In[2] : y_limits(Y, 0.05)
	Out[2] : (-5.5, 5.5)

	"""
	ampli = np.max(a)-np.min(a)
	lower = np.min(a)-ampli*marge
	upper = np.max(a)+ampli*marge
	return lower, upper

def show_height(x, y, z, colormap=None, marge=0.01, fig_sz=(10, 8), msz=25,
			   title=None, cent_cmap=True):
	"""
	Fonction to show the triangular mesh and the height of the dots.

	Parameters
	----------
	x : numpy.ndarray
		X-axis positions of the nodes.
	y : numpy.ndarray
		Y-axis positions of the nodes.
	z : numpy.ndarray
		A 1-dimensional array that store the height for all nodes.
	colormap : str, optional
		Colormap to use for the height of the dots. The default is None.
	marge : float, optional
		A float in the range [0, 1]. It is used to calculate the space between
		the dots and the bound of the figure trhough xlim and ylim function of
		matplotlib.pyplot package. The default is 0.01.
	fig_sz : tuple, optional
		A 1-dimensional tuple that store the size of the figure. It will be
		used trhough the function figure from the matplotlib.pyplot package.
		The default is (10, 8).
	msz : flot, optional
		Size of the dots for the scatter function. The default is 25.
	title : str, optional
		Title to put if asked. The default is None.

	Returns
	-------
	None.

	Exemple
	-------
	In[0] : X, Y = create_mesh(151, 'square')
	In[1] : h = np.random.normal(0, 1, len(X))
	In[2] : show_height(x, y, h, 'jet', 0.02, (12, 10), 15, 'random heigth',
						True) 
	Out[2] : matplotlib.pyplot object

	"""
	if cent_cmap:
		limits = np.max(np.abs(z))

	inf_x, sup_x = y_limits(x, marge)
	inf_y, sup_y = y_limits(y, marge)

	plt.figure(figsize=fig_sz)
	if type(title) == str:
		plt.title(title, fontsize=12)

	if cent_cmap:
		if type(colormap) != type(None):
			plt.scatter(x, y, msz, z, zorder=3, cmap=colormap, vmin=-limits,
						vmax=limits)
		else:
			plt.scatter(x, y, msz, z, zorder=3, vmin=-limits, vmax=limits)

	else:
		if type(colormap) != type(None):
			plt.scatter(x, y, msz, z, zorder=3, cmap=colormap)
		else:
			plt.scatter(x, y, msz, z, zorder=3)

	plt.axis('equal')
	plt.colorbar(shrink=0.9, aspect=25, pad=0.01)
	plt.xlim(inf_x, sup_x)
	plt.ylim(inf_y, sup_y)
	plt.show()

def plot_maze_connect(x, y, connections, z=None, marge=0.02, title=None,
					fig_sz=(12, 12), solution=None):
	"""
	Fonction to show the maze created with the triangular mesh. It is possible
	to plot the gradient, and the solution.

	Parameters
	----------
	x : numpy.ndarray
		X-axis positions of the nodes.
	y : numpy.ndarray
		Y-axis positions of the nodes.
	connections : numpy.ndarray
		A 2-dimensions array that store the interconnections between the dots.
	z : numpy.ndarray, optional
		A 1-dimensional array that store the distance (in nod number) for all
		the nodes from the node  at index ID_st_node. The default is None.
	marge : float, optional
		A float in the range [0, 1]. It is used to calculate the space between
		the dots and the bound of the figure trhough xlim and ylim function of
		matplotlib.pyplot package. The default is 0.02.
	fig_sz : tuple, optional
		A 1-dimensional tuple that store the size of the figure. It will be
		used trhough the function figure from the matplotlib.pyplot package.
		The default is (12, 12).
	solution : numpy.ndarray, optional
		A 2-dimensional array that store a path between two nods. The default
		is None.
	title : str, optional
		Title to put if asked. The default is None.

	Returns
	-------
	None.

	Exemple
	-------
	In[0] : x, y = create_mesh(51, 'square')
	In[2] : adic = groupe_by(x, y)
	In[3] : vdm = maze_fusion(adic)
	In[4] : gradient = Dijkstra_triangular_mesh(x, y, vdm, 0)
	In[5] : id_st = len(x)-1
	In[6] : pid = optimal_path(gradient, vdm, id_st)
	In[7] : sol = np.array([x[pid], y[pid]]).T
	In[8] : plot_maze_connect(x, y, vdm, gradient, 0.01, solution=sol) 
	Out[8] : matplotlib.pyplot object

	Note
	----
	Due to the two for loop, it have a long calculation time.

	"""
	inf_x, sup_x = y_limits(x, marge)
	inf_y, sup_y = y_limits(y, marge)

	plt.figure(figsize=fig_sz)
	if type(title) == str:
		plt.title(title, fontsize=12)

	if type(z) == np.ndarray:
		plt.scatter(x, y, 20, z, zorder=3, cmap='jet')

	else:
		plt.plot(x, y, '.', zorder=3)
	for i in range(len(connections)):
		for j in range(len(connections[i]-1)):
			plt.plot([x[connections[i][0]], x[connections[i][j]]],
					 [y[connections[i][0]], y[connections[i][j]]],
					 'k', zorder=2, lw=0.5)

	if type(solution) == np.ndarray:
		plt.plot(solution[:, 0], solution[:, 1], 'm', zorder=3, lw=2)

	plt.axis('equal')
	plt.xlim(inf_x, sup_x)
	plt.ylim(inf_y, sup_y)
	plt.show()
