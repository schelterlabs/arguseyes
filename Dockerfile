# Start with Python 3.9
FROM python:3.9

# Install Graphviz
RUN apt update && apt install -y graphviz graphviz-dev

# Install ArgusEyes
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
RUN pip install -U pip && pip install git+https://github.com/schelterlabs/arguseyes.git@sigmod-demo

# Run CLI
ENTRYPOINT [ "python", "-m", "arguseyes" ]
