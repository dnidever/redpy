#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='redpy',
      version='1.0',
      description='Python reduction software',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/redpy',
      packages=find_packages(exclude=["tests"]),
      requires=['numpy','astropy','dlnpyutils'],
#      include_package_data=True,
)
