dbscan
=====

Density Based Spatial Clustering of Applications with python plotting

Shun Xu <alwintui@gmail.com>

July 5, 2014

Setup
---------

```Bash
	python setup.py install
```

Usages
----------
```Python
	import dbscan
	dbscan.dbscan(m, eps, min_points)
```

**or**
```Bash
	python -m dbscan.dbscan eps minpts data-file  col0 col1 col2 ...
```

```Bash
	dbscan.py eps minpts data-file  col0 col1 col2 ...
```

Examples
--------------
```Bash
	python -m dbscan.dbscan 0.04 5 test1500.txt 0 1 2
	python -m dbscan.dbscan 0.02 6 test1500.txt 0 1
```
