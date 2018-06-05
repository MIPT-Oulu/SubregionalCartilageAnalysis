#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = []

setup_requirements = []

setup(
    author="Egor Panfilov",
    author_email='egor.v.panfilov@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Sub-regional morphological analysis of knee cartilage from MRI",
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    name='scartan',
    packages=find_packages(include=['scartan']),
    setup_requires=setup_requirements,
    url='https://github.com/MIPT-Oulu/SubregionalCartilageAnalysis',
    version='0.1.0',
    zip_safe=False,
)
