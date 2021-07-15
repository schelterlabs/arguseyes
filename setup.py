"""
For 'python setup.py develop' and 'python setup.py test'
"""
import os
from setuptools import setup, find_packages

ROOT = os.path.dirname(__file__)

setup(
    name="arguseyes",
    version="0.0.1.dev0",
    author='Sebastian Schelter',
    author_email='s.schelter@uva.nl',
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "mlinspect[dev] @ git+https://github.com/stefan-grafberger/mlinspect.git@"
        "ea06a560e32a16b579823d9e7de36a038cf08908",
        "importnb==0.6.2",
        "matplotlib==3.4.2",
        "tensorflow==2.5.0",
        "mlflow==1.18"
    ],
    license='Apache License 2.0',
    python_requires='==3.9.*',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9'
    ]
)
