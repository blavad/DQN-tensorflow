#!/usr/bin/env python

import imp
from setuptools import setup, find_packages

setup(
    name='dqn-bc',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['tf-explain','tensorflow==2.0.0-beta1', 'opencv','numpy'],
)
