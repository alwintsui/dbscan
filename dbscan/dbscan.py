#!/usr/bin/env python
# -*- coding: utf-8 -*-

help_str = """dbscan: Density Based Spatial Clustering of Applications with python plotting

Usages:
dbscan.py eps minpts data-file  col0 col1 col2 ...

Examples:
dbscan.py 0.04 5 test1500.txt 0 1 2
dbscan.py 0.02 6 test1500.txt 0 1 

Shun Xu <alwintui@gmail.com>
July 5,2014"""

import numpy as np

UNCLASSIFIED = None
NOISE = 0

nblist = {}	
def _build_cache(m, eps):
	''' build distance cache.
        each vector will compute the distance with others
        distance from others are sorted ASC '''
	global nblist
	n_points = m.shape[0]	
	# init								
	i = 0
	while i < n_points:
		nblist[i] = []
		i += 1
	# build
	i = 0
	while i < n_points - 1:
		j = i + 1
		while j < n_points:
			# _eps_neighborhood, or np.sqrt(np.power(m[i, :] - m[j, :], 2).sum()) < eps
			if np.linalg.norm(m[i, :] - m[j, :]) < eps: 
				nblist[i].append(j)
				nblist[j].append(i)
			j += 1
		i += 1
	return
        
def _expand_cluster(classifications, point_id, cluster_id, eps, min_points):
	global nblist
	seeds = nblist[point_id][:]  # deepcopy
	if len(seeds) < min_points:
		classifications[point_id] = NOISE
		return False
	else:
		classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = nblist[current_point]
            if len(results) >= min_points:
                for result_point in results:
                    if classifications[result_point] is UNCLASSIFIED or \
                       classifications[result_point] is NOISE:
                        if classifications[result_point] is UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True

def dbscan(m, eps, min_points):
	"""Implementation of Density Based Spatial Clustering of Applications with Noise
    See https://en.wikipedia.org/wiki/DBSCAN
    
    scikit-learn probably has a better implementation
    
    Uses Euclidean Distance as the measure
    
    Inputs:
    m - A matrix whose columns are feature vectors
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    An array with either a cluster id number or dbscan.NOISE (None) for each
    column vector in m.
    """
	global nblist
	if not nblist:
		_build_cache(m, eps)
	n_points = m.shape[0]
	cluster_id = 1
	classifications = [UNCLASSIFIED] * n_points
	for point_id in range(0, n_points):
		if classifications[point_id] is UNCLASSIFIED:
			if _expand_cluster(classifications, point_id, cluster_id, eps, min_points):
				cluster_id = cluster_id + 1
	return classifications

def group_sep(cat):
	"""group all continuing points in neibors ignoring noise by index 0"""
	s = 0
	g = False
	l = len(cat)
	i = 0
	gs = []
	while i < l:
		if cat[i] == 0:
			i += 1
			continue
		if g == False:
			g = cat[i]
		if g != cat[i]:
			gs.append((g, s, i, i - s))
			s = i
			g = cat[i]
		i += 1
	gs.append((g, s, i, i - s))
	return gs

def plt_m3cat(m, cat, fpre):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import axes3d
	fig = plt.figure(dpi=100)
	ax = fig.gca(projection='3d')
	cm = plt.get_cmap('Reds')
	plt.hold(True)
	colors = ['r', 'g', 'b', 'c', 'm', 'y']
	marks = ['^', 'o', 'v', '<', '>', '.', 's']
	ca = np.array(cat, dtype=np.int32)
	cmax = np.amax(ca)
	ci = ca == 0
	ax.scatter(m[ci, 0], m[ci, 1], m[ci, 2], marker='.', \
				c='k', s=15, edgecolors='None')
	for c in xrange(1, cmax + 1):
		ci = ca == c
		ax.scatter(m[ci, 0], m[ci, 1], m[ci, 2], marker=marks[c % len(marks)], \
				c=colors[c % len(colors)], s=15, edgecolors='None')
	ax.set_xlabel('x', fontsize=12)
	ax.set_ylabel('y', fontsize=12)
	ax.set_zlabel('z', fontsize=12)	
   	# plt.show()
   	of = "%sn%dg%d.png" % (fpre, m.shape[0], cmax)
   	print of
   	plt.savefig(of)
   	plt.close()
def plt_m2cat(m, cat, fpre):
	import matplotlib.pyplot as plt
	ca = np.array(cat, dtype=np.int32)
	cmax = np.amax(ca)
	plt.hold(True)
	colors = ['r', 'g', 'b', 'c', 'm', 'y']
	marks = ['^', 'o', 'v', '<', '>', '.', 's']
	ci = ca == 0
	plt.plot(m[ci, 0], m[ci, 1], 'k.')
	for c in xrange(1, cmax + 1):
		ci = ca == c
		plt.plot(m[ci, 0], m[ci, 1], marker=marks[c % len(marks)], c=colors[c % len(colors)])
	plt.xlabel('x', fontsize=12)
	plt.ylabel('y', fontsize=12)
   	# plt.show()
   	of = "%sn%dg%d.png" % (fpre, m.shape[0], cmax)
   	print of
   	plt.savefig(of)
   	plt.close()

if __name__ == '__main__':
	from sys import argv, exit
	from os.path import splitext
	import time
	if len(argv) < 6:
		print help_str
		exit(-1)
	eps = float(argv[1])  # 0.04
	min_points = int(argv[2])  # 3
	fpre, ext = splitext(argv[3])
	fpre = "%s_e%gp%dc%s" % (fpre, eps, min_points, "x".join(argv[4:]))
	# load and norm all columns
	uc = np.array(argv[4:], dtype=np.int32)
	m = np.loadtxt(argv[3], usecols=uc)
	amax = np.amax(m, axis=0)
	amin = np.amin(m, axis=0)
	anorm = (m - amin) / (amax - amin)
 	# dbscan
	startTime = time.clock()
	print "starting dbscan ..."
	cat = dbscan(anorm, eps, min_points)
	print "completed with elasped time (sec.):", time.clock() - startTime
	# print nb size, class
	n_points = m.shape[0]
	i = 0
	while i < n_points:
		print i, len(nblist[i]), cat[i]
		i += 1
	grp = group_sep(cat)
	print grp

	if m.shape[1] == 2:
		plt_m2cat(m, cat, fpre)
	elif m.shape[1] == 3:
		plt_m3cat(m, cat, fpre)
	print "finished with elasped time (sec.):", time.clock() - startTime
