#!/usr/bin/env python

from distutils.core import setup
#from setuptools import setup

setup(name="dbscan",
	  version="1.0",
	  description="Density Based Spatial Clustering of Applications with python plotting",
	  author = "Shun Xu",
	  author_email="alwintsui@gmail.com",
	  packages=['dbscan'],
	  package_dir={'dbscan': 'dbscan'})
