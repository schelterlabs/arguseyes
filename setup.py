import os
from setuptools import setup, find_packages

ROOT = os.path.dirname(__file__)

with open(os.path.join(ROOT, "requirements.txt")) as f:
    required = f.read().splitlines()

setup(
    name="arguseyes",
    version="0.0.1.dev0",
    author='Sebastian Schelter',
    author_email='s[DOT]schelter[AT]uva[DOT]nl',
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    license='GNU General Public License v3 (GPLv3)',
    python_requires='==3.9.*',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9'
    ]
)
