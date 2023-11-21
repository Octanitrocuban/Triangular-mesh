# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 22:49:21 2023
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import mesh
import plot

to_do = ['maze_connections']


if 'maze' in to_do:
	n = 20
	m = 'circle'
	s = 'optimal'
	e = 'explore'
	c = False
	if m == 'circle':
		id0, id1 = mesh.id_start_end_nods(n)
		X, Y = mesh.create_mesh(n, m)
		adic = mesh.groupe_by(X, Y)
		if e == 'fusion':
			vdm = mesh.maze_fusion(adic)
		elif e == 'explore':
			vdm = mesh.maze_exploration(adic)

		if c:
			vdm = mesh.complexification(adic, vdm)

		if s == 'hand':
			pid = mesh.hand_solving(vdm, X, Y, 'left', id0, id1, 0)
		elif s == 'optimal':
			d = mesh.Dijkstra_triangular_mesh(vdm, id1)
			pid = mesh.optimal_path(d, vdm, id0)

	elif m == 'square':
		X, Y = mesh.create_mesh(n, m)
		adic = mesh.groupe_by(X, Y)
		if e == 'fusion':
			vdm = mesh.maze_fusion(adic)
		elif e == 'explore':
			vdm = mesh.maze_exploration(adic, len(X)//2)

		if c:
			vdm = mesh.complexification(adic, vdm)

		if s == 'hand':
			pid = mesh.hand_solving(vdm, X, Y, 'left', 0, len(X)-1, 0)
		elif s == 'optimal':
			d = mesh.Dijkstra_triangular_mesh(vdm, len(X)-1)
			pid = mesh.optimal_path(d, vdm, 0)

	sol = np.array([X[pid], Y[pid]]).T
	plot.plot_maze_connect(X, Y, vdm, None, 0.01, solution=sol)


if 'height' in to_do:
	m = 2
	s = 'square'
	if s == 'square':
		n = 151
		X, Y = mesh.create_mesh(n, s)
		if m == 1:
			h = n*np.random.normal(0, 1, len(X))
		elif m == 2:
			adic = mesh.groupe_by(X, Y)
			vdm = mesh.maze_fusion(adic)
			dc1 = mesh.Dijkstra_triangular_mesh(vdm, 0)
			dc2 = mesh.Dijkstra_triangular_mesh(vdm, len(X)-1)
			dc3 = mesh.Dijkstra_triangular_mesh(vdm, n-1)
			h = dc1+dc2+dc3+ np.random.normal(0, 1, len(X))
			h = h-np.min(h)
			h = (h-np.max(h)/2)/2

	elif s == 'circle':
		n = 51
		id0, id1 = mesh.id_start_end_nods(n)
		X, Y = mesh.create_mesh(n, s)
		if m == 1:
			h = n*np.random.normal(0, 1, len(X))
		elif m == 2:
			adic = mesh.groupe_by(X, Y)
			vdm = mesh.maze_fusion(adic)
			dc1 = mesh.Dijkstra_triangular_mesh(vdm, id0)
			dc2 = mesh.Dijkstra_triangular_mesh(vdm, id1)
			h = dc1+dc2+ n*np.random.normal(0, 1, len(X))
			h = h-np.min(h)
			h = (h-np.max(h)/2)/2

	nsm = 4 ; ks = 5
	plot.show_height(X, Y, h, 'jet', msz=20, title='Raw', cent_cmap=False)
	for i in range(nsm):
		h = h+ np.random.normal(0, 1, len(X))/(i+2)**2
		h = mesh.lin_smooth(X, Y, h, ks)
		plot.show_height(X, Y, h, 'jet', msz=20,
					   title='smooth: '+str(i+1)+'/'+str(nsm),
					   cent_cmap=False)


if 'maze_connections' in to_do:
	# In this part, we will investigate the interconection of the nodes
	# created by the maze
	m = 'square'
	if m == 'circle':
		n = 29
		X, Y = mesh.create_mesh(n, m)
	elif m == 'square':
		n = 51
		X, Y = mesh.create_mesh(n, m)
	adic = mesh.groupe_by(X, Y)
	len_fusion = []
	len_explor = []
	for i in tqdm(range(100)):
		vdmf = mesh.maze_fusion(adic)
		vdme = mesh.maze_exploration(adic)
		tf = []
		te = []
		for j in range(len(vdmf)):
			tf.append(len(vdmf[j])-1)
			te.append(len(vdme[j])-1)

		len_fusion.append(tf)
		len_explor.append(te)

	len_fusion = np.array(len_fusion)
	len_explor = np.array(len_explor)
	
	distrib_f = []
	distrib_e = []
	for i in range(len(len_fusion)):
		distrib_f.append(mesh.comptage(len_fusion[i]))
		distrib_e.append(mesh.comptage(len_explor[i]))

	distrib_f = np.array(distrib_f)
	distrib_e = np.array(distrib_e)

	mean_f = np.mean(distrib_f, axis=0)
	max_f = np.max(distrib_f, axis=0)
	min_f = np.min(distrib_f, axis=0)

	mean_e = np.mean(distrib_e, axis=0)
	max_e = np.max(distrib_e, axis=0)
	min_e = np.min(distrib_e, axis=0)

	plt.figure(figsize=(10, 5))
	plt.grid(True, zorder=1)
	plt.plot(np.arange(6)+1, mean_f, 'd', color='steelblue',
			 label='mean of fusion method', zorder=4)
	plt.plot(np.arange(6)+1, max_f, color='steelblue', label='min-max range')
	plt.plot(np.arange(6)+1, min_f, color='steelblue')
	plt.plot(np.arange(6)+1, mean_e, 'd', color='orange',
		     label='mean of exploration method', zorder=4)
	plt.plot(np.arange(6)+1, max_e, color='orange', label='min-max range')
	plt.plot(np.arange(6)+1, min_e, color='orange')
	plt.legend(fontsize=13)
	plt.xlabel('Nombre de connexions par noeud')
	plt.ylabel('Nombre de noeud ayant ce nombre de connexion')
	plt.show()
