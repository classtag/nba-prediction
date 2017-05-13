#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0', 'numpy', 'pandas', 'sklearn'
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='nba_prediction',
    version='0.1.0',
    description="Build a prediction model for nba next season n via machine learning. Package include sklearn",
    long_description=readme + '\n\n' + history,
    author="Victor An",
    author_email='anduo@qq.com',
    url='https://github.com/classtag/nba_prediction',
    packages=[
        'nba_prediction',
    ],
    package_dir={'nba_prediction':
                 'nba_prediction'},
    entry_points={
        'console_scripts': [
            'nba_prediction=nba_prediction.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='nba_prediction',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
