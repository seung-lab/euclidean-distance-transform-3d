import os
import setuptools

import numpy as np

# NOTE: If edt.cpp does not exist:
# cython -3 --fast-fail -v --cplus edt.pyx

setuptools.setup(
  setup_requires=['pbr'],
  extras_require={
    ':python_version == "2.7"': ['futures'],
    ':python_version == "2.6"': ['futures'],
  },
  ext_modules=[
    setuptools.Extension(
      'edt',
      sources=[ 'edt.cpp' ],
      language='c++',
      include_dirs=[ np.get_include(), 'threadpool.h' ],
      extra_compile_args=[
        '-std=c++11', '-O3', '-ffast-math', '-pthread'
      ]
    ),
  ],
  pbr=True)