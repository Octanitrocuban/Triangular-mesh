# -*- coding: utf-8 -*-
"""
Module to test the functions to create and analyze hexagonal mesh.
"""
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
import mesh
import plot

to_do = ['maze_stats']

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
	n_nodes = np.array([11, 21, 31, 41, 51, 61, 71, 81], dtype=int)
	method = 'square'
	c = 0
	stats = []
	temps = []
	for sz in n_nodes:
		temps.append([[], [], []])
		stats.append([])
		reseau = mesh.create_mesh(sz, method)
		for i in tqdm(range(1000)):
			sts = np.zeros((3, 7))
			t0 = time()
			adic = mesh.groupe_by(reseau)
			maze_fus = mesh.maze_fusion(adic)
			temps[c][0].append(time()-t0)

			t0 = time()
			adic = mesh.groupe_by(reseau)
			maze_exp = mesh.maze_exploration(adic)
			temps[c][1].append(time()-t0)

			t0 = time()
			maze_kur = mesh.kurskal_maze(reseau)
			temps[c][2].append(time()-t0)

			if len(maze_fus) != len(maze_exp):
				raise

			if len(maze_fus) != len(maze_kur):
				raise

			for j in range(len(maze_fus)):
				sts[0, len(maze_fus[j])-1] += 1
				sts[1, len(maze_exp[j])-1] += 1
				sts[2, len(maze_kur[j])-1] += 1

			stats[c].append(sts)

		c += 1

	stats = np.transpose(np.array(stats), (0, 2, 1, 3))
	temps = np.transpose(np.array(temps), (0, 2, 1))

	np.save('temps.npy', temps, allow_pickle=True)
	np.save('stats.npy', stats, allow_pickle=True)

	plt.figure(figsize=(14, 8))
	plt.title('Time consumption', fontsize=18)
	plt.grid(True, zorder=1)
	plt.plot(n_nodes, np.mean(temps[:, :, 0], axis=1), 'b-', zorder=5,
			 label='fusion')

	violonplt1 = plt.violinplot(temps[:, :, 0].T, n_nodes, widths=3,
								showmeans=True, showextrema=True,
								showmedians=True)

	for pc in violonplt1['bodies']:
	    pc.set_color('b')
	
	violonplt1['cmeans'].set_color([0, 0, 1])
	violonplt1['cmins'].set_color([0, 0, 1])
	violonplt1['cmaxes'].set_color([0, 0, 1])
	violonplt1['cbars'].set_color([0, 0, 1])
	violonplt1['cmedians'].set_color([0, 0, 1])
	plt.plot(n_nodes, np.mean(temps[:, :, 1], axis=1), 'g-', zorder=5,
			 label='random walk')

	violonplt2 = plt.violinplot(temps[:, :, 1].T, n_nodes, widths=3,
								showmeans=True, showextrema=True,
								showmedians=True)

	for pc in violonplt2['bodies']:
	    pc.set_color('g')
	
	violonplt2['cmeans'].set_color([0, .8, 0])
	violonplt2['cmins'].set_color([0, .8, 0])
	violonplt2['cmaxes'].set_color([0, .8, 0])
	violonplt2['cbars'].set_color([0, .8, 0])
	violonplt2['cmedians'].set_color([0, .8, 0])
	plt.plot(n_nodes, np.mean(temps[:, :, 2], axis=1), 'r-', zorder=5,
			 label='kurskal')

	violonplt3 = plt.violinplot(temps[:, :, 2].T, n_nodes, widths=3,
								showmeans=True, showextrema=True,
								showmedians=True)

	for pc in violonplt3['bodies']:
	    pc.set_color('r')

	violonplt3['cmeans'].set_color([1, 0, 0])
	violonplt3['cmins'].set_color([1, 0, 0])
	violonplt3['cmaxes'].set_color([1, 0, 0])
	violonplt3['cbars'].set_color([1, 0, 0])
	violonplt3['cmedians'].set_color([1, 0, 0])
	plt.xticks(n_nodes, n_nodes, fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlabel('Width size of the mazes', fontsize=15)
	plt.ylabel('Time (s)', fontsize=15)
	plt.legend(fontsize=15, loc='upper left', title='Methods',
				title_fontsize=15)

	plt.savefig('time_contruction_methods.png')
	plt.show()

	num_nodes = np.sum(stats, axis=3)[:, :, :, np.newaxis]
	pdf = stats/num_nodes
	for i in range(1, 7):
	    plt.figure(figsize=(14, 8))
	    plt.title('Distribution of node with '+str(i)+' connections',
					fontsize=18)

	    plt.grid(True, zorder=1)
	    plt.plot(n_nodes, np.mean(pdf[:, 0, :, i], axis=1), 'b-', zorder=5,
				 label='fusion')

	    violonplt1 = plt.violinplot(pdf[:, 0, :, i].T, n_nodes, widths=3,
									showmeans=True, showextrema=True,
									showmedians=True)

	    for pc in violonplt1['bodies']:
	        pc.set_color('b')

	    violonplt1['cmeans'].set_color([0, 0, 1])
	    violonplt1['cmins'].set_color([0, 0, 1])
	    violonplt1['cmaxes'].set_color([0, 0, 1])
	    violonplt1['cbars'].set_color([0, 0, 1])
	    violonplt1['cmedians'].set_color([0, 0, 1])

	    plt.plot(n_nodes, np.mean(pdf[:, 1, :, i], axis=1), 'g-', zorder=5,
				 label='random walk')

	    violonplt2 = plt.violinplot(pdf[:, 1, :, i].T, n_nodes, widths=3,
									showmeans=True, showextrema=True,
									showmedians=True)

	    for pc in violonplt2['bodies']:
	        pc.set_color('g')

	    violonplt2['cmeans'].set_color([0, .6, 0])
	    violonplt2['cmins'].set_color([0, .6, 0])
	    violonplt2['cmaxes'].set_color([0, .6, 0])
	    violonplt2['cbars'].set_color([0, .6, 0])
	    violonplt2['cmedians'].set_color([0, .6, 0])
	    plt.plot(n_nodes, np.mean(pdf[:, 2, :, i], axis=1), 'r-', zorder=5,
				 label='kurskal')

	    violonplt3 = plt.violinplot(pdf[:, 2, :, i].T, n_nodes, widths=3,
									showmeans=True, showextrema=True,
									showmedians=True)

	    for pc in violonplt3['bodies']:
	        pc.set_color('r')

	    violonplt3['cmeans'].set_color([1, 0, 0])
	    violonplt3['cmins'].set_color([1, 0, 0])
	    violonplt3['cmaxes'].set_color([1, 0, 0])
	    violonplt3['cbars'].set_color([1, 0, 0])
	    violonplt3['cmedians'].set_color([1, 0, 0])
	    plt.xticks(n_nodes, n_nodes, fontsize=15)
	    plt.yticks(fontsize=15)
	    plt.xlabel('Width size of the mazes', fontsize=15)
	    plt.ylabel('Distribution pdf (nodes/tot nodes)', fontsize=15)
	    plt.legend(fontsize=15, loc='upper right', title='Methods',
				   title_fontsize=15, ncol=3)

	    plt.savefig('distribution_of_connections_'+str(i)+'.png')
	    plt.show()
