import setuptools
import sys

import numpy as np

# NOTE: If edt.cpp does not exist:
# cython -3 --fast-fail -v --cplus edt.pyx

extra_compile_args = [
  '-std=c++11', '-O3', '-ffast-math', '-pthread'
]
if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

setuptools.setup(
  setup_requires=['pbr'],
  python_requires="~=3.6", # >= 3.6 < 4.0
  ext_modules=[
    setuptools.Extension(
      'edt',
      sources=[ 'edt.cpp' ],
      language='c++',
      include_dirs=[ np.get_include() ],
      extra_compile_args=extra_compile_args,
    ),
  ],
  long_description_content_type='text/markdown',
  pbr=True
)