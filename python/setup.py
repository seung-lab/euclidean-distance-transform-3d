import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

# NOTE: If edt.cpp does not exist:
# cython -3 --fast-fail -v --cplus edt.pyx

extra_compile_args = []
if sys.platform == 'win32':
  extra_compile_args += [
    '/std:c++11', '/O2'
  ]
else:
  extra_compile_args += [
    '-std=c++11', '-O3', '-ffast-math', '-pthread'
  ]

if sys.platform == 'darwin':
  extra_compile_args += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]


setuptools.setup(
  setup_requires=['pbr', 'cython'],
  python_requires=">=3.7,<4",
  ext_modules=[
    setuptools.Extension(
      'edt',
      sources=[ 'edt.pyx' ],
      language='c++',
      include_dirs=[ NumpyImport() ],
      extra_compile_args=extra_compile_args,
    ),
  ],
  long_description_content_type='text/markdown',
  pbr=True
)