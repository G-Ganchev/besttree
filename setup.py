from Cython.Build import cythonize
from setuptools import Extension ,setup

# define an extension that will be cythonized and compiled
ext = Extension(name="class_tree", sources=["classification_tree.pyx"])
setup(name="class_tree",ext_modules=cythonize(ext))