FROM quay.io/pypa/manylinux2014_x86_64 
MAINTAINER William Silversmith

ADD . /edt
WORKDIR "/edt/python"

ENV CXX "g++"

RUN rm -rf *.so build __pycache__ dist 

RUN /opt/python/cp36-cp36m/bin/pip3.6 install pip --upgrade
RUN /opt/python/cp36-cp36m/bin/pip3.6 install oldest-supported-numpy
RUN /opt/python/cp36-cp36m/bin/pip3.6 install -r requirements_dev.txt
RUN /opt/python/cp36-cp36m/bin/python3.6 setup.py develop
RUN /opt/python/cp36-cp36m/bin/python3.6 -m pytest -v -x automated_test.py

RUN /opt/python/cp37-cp37m/bin/pip3.7 install pip --upgrade
RUN /opt/python/cp37-cp37m/bin/pip3.7 install oldest-supported-numpy
RUN /opt/python/cp37-cp37m/bin/pip3.7 install -r requirements_dev.txt
RUN /opt/python/cp37-cp37m/bin/python3.7 setup.py develop
RUN /opt/python/cp37-cp37m/bin/python3.7 -m pytest -v -x automated_test.py

RUN /opt/python/cp38-cp38/bin/pip3.8 install pip --upgrade
RUN /opt/python/cp38-cp38/bin/pip3.8 install oldest-supported-numpy
RUN /opt/python/cp38-cp38/bin/pip3.8 install -r requirements_dev.txt 
RUN /opt/python/cp38-cp38/bin/python3.8 setup.py develop
RUN /opt/python/cp38-cp38/bin/python3.8 -m pytest -v -x automated_test.py

RUN /opt/python/cp39-cp39/bin/pip3.9 install pip --upgrade
RUN /opt/python/cp39-cp39/bin/pip3.9 install oldest-supported-numpy pytest
RUN /opt/python/cp39-cp39/bin/pip3.9 install -r requirements_dev.txt 
RUN /opt/python/cp39-cp39/bin/python3.9 setup.py develop
RUN /opt/python/cp39-cp39/bin/python3.9 -m pytest -v automated_test.py

RUN /opt/python/cp36-cp36m/bin/python3.6 setup.py bdist_wheel
RUN /opt/python/cp37-cp37m/bin/python3.7 setup.py bdist_wheel
RUN /opt/python/cp38-cp38/bin/python3.8 setup.py bdist_wheel
RUN /opt/python/cp39-cp39/bin/python3.9 setup.py bdist_wheel

RUN for whl in `ls dist/*.whl`; do auditwheel repair $whl --plat manylinux2014_x86_64; done
