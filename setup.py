#!/usr/bin/env python

from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='pyautoweka',
      version='.1',
      description='AutoWeka for python',
      author='Tobias Domhan',
      author_email='tdomhan@gmail.com',
      url='http://www.cs.ubc.ca/labs/beta/Projects/autoweka/',
      packages=['pyautoweka'],
      #package_data={"pyautoweka": ["./java/weka.jar"]},
      include_package_data = True,
      eager_resources=["pyautoweka/weka.jar"],
      long_description=read('README.md'),
      requires=[
          'lxml',
          'numpy'
      ],
     )
