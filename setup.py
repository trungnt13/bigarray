#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

import pip
from setuptools import find_packages, setup

_BIGARRAY_VERSION = '0.1.0'

# ===========================================================================
# Main
# ===========================================================================
with open('README.md') as readme_file:
  readme = readme_file.read()

author = 'Trung Ngo.T.'

requirements = ["numpy>=1.15.0", "six>=1.12.0"]

setup(
    author=author,
    author_email='trung@imito.ai',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache License',
        'Natural Language :: English',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
    ],
    description=
    "Fast and scalable array for machine learning and artificial intelligence",
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=['pip>=19.0'],
    install_requires=requirements,
    license="Apache license",
    include_package_data=True,
    keywords='bigarray',
    name='bigarray',
    packages=find_packages(),
    test_suite='tests',
    url='https://github.com/trungnt13/bigarray',
    version=_BIGARRAY_VERSION,
    zip_safe=False,
)
