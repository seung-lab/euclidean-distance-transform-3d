import os
import platform
import setuptools
import sys

class NumpyImport:
  def __repr__(self):
    import numpy as np

    return np.get_include()

  __fspath__ = __repr__

# NOTE: If edt.cpp does not exist:
# cython -3 --fast-fail -v --cplus edt.pyx

extra_compile_args_nd = []
extra_compile_args_legacy = []
machine = platform.machine().lower()
is_x86 = machine in ("x86_64", "amd64")
enable_native = os.environ.get("EDT_MARCH_NATIVE", "1").strip().lower()
use_native = enable_native not in ("0", "false", "no", "off", "")
building_wheel = any(arg.startswith("bdist_wheel") or arg == "--wheel" for arg in sys.argv)
if building_wheel:
  use_native = False
if sys.platform == 'win32':
  # /wd4551: suppress "function call missing argument list" from Cython-generated code
  # (Cython emits `(void) func_name;` to silence unused-function warnings)
  common_win = ['/std:c++17', '/O2', '/wd4551']
  extra_compile_args_nd += common_win
  extra_compile_args_legacy += common_win
else:
  extra_compile_args_nd += [
    '-std=c++17',
    '-O3', '-ffast-math', '-fno-finite-math-only', '-fno-unsafe-math-optimizations',
    '-fno-math-errno', '-fno-trapping-math',
    '-flto', '-DNDEBUG', '-pthread'
  ]
  if is_x86 and use_native:
    extra_compile_args_nd += ['-march=native', '-mtune=native']

  # Match upstream legacy flags to minimize divergence.
  extra_compile_args_legacy += [
    '-std=c++17', '-O3', '-ffast-math', '-pthread'
  ]

if sys.platform == 'darwin':
  extra_compile_args_nd += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]
  extra_compile_args_legacy += [ '-stdlib=libc++', '-mmacosx-version-min=10.9' ]

# Add extra_link_args for LTO if not Windows (ND only)
extra_link_args_nd = []
extra_link_args_legacy = []
if sys.platform != 'win32':
  extra_link_args_nd += ['-flto']


extensions = [
  # Main EDT module (graph-first ND v2 architecture)
  setuptools.Extension(
    'edt',
    sources=['src/edt.pyx'],
    language='c++',
    include_dirs=['src', str(NumpyImport())],
    extra_compile_args=extra_compile_args_nd,
    extra_link_args=extra_link_args_nd,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
  ),
  # Legacy upstream implementation (for comparison)
  setuptools.Extension(
    'edt_legacy',
    sources=['legacy/edt.pyx'],
    language='c++',
    include_dirs=['legacy', str(NumpyImport())],
    extra_compile_args=extra_compile_args_legacy,
    extra_link_args=extra_link_args_legacy,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
  ),
]

setuptools.setup(
  setup_requires=['cython', 'setuptools_scm'],
  python_requires=">=3.8,<4",
  use_scm_version=True,
  ext_modules=extensions,
  long_description_content_type='text/markdown',
)
