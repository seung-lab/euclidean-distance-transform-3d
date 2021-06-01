FROM quay.io/pypa/manylinux1_x86_64
LABEL maintainer="William Silversmith"

ADD . /edt
WORKDIR "/edt/python"

ENV CXX "g++"

RUN rm -rf *.so build __pycache__ dist 

RUN /opt/python/cp36-cp36m/bin/pip3.6 install pip --upgrade
RUN /opt/python/cp37-cp37m/bin/pip3.7 install pip --upgrade
RUN /opt/python/cp38-cp38/bin/pip3.8 install pip --upgrade 

RUN /opt/python/cp36-cp36m/bin/pip3.6 install oldest-supported-numpy
RUN /opt/python/cp37-cp37m/bin/pip3.7 install oldest-supported-numpy
RUN /opt/python/cp38-cp38/bin/pip3.8 install oldest-supported-numpy

RUN /opt/python/cp36-cp36m/bin/python3.6 setup.py bdist_wheel
RUN /opt/python/cp37-cp37m/bin/python3.7 setup.py bdist_wheel
RUN /opt/python/cp38-cp38/bin/python3.8 setup.py bdist_wheel

RUN for whl in `ls dist/*.whl`; do auditwheel repair $whl; done
