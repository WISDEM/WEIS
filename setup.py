# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import os

from setuptools import setup, find_packages

import versioneer

REQUIRED = [
    "matplotlib",
    "numpy",
    "pytest",
    "scipy",
    "pyYAML",
    "seaborn",
    "fatpack",
]

ROOT = os.path.abspath(os.path.dirname(__file__))
with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="pCrunch",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="IO and Post Processing Interface for OpenFAST Results.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=["Jake Nunemaker", "Nikhar Abbas"],
    author_email=["Jake.Nunemaker@nrel.gov", "nikhar.abbas@nrel.gov"],
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*"]
    ),
    package_data={"": ["*.out", ".outb"]},
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "pyYAML",
        "seaborn",
        "fatpack",
    ],
    extras_require={
        "dev": [
            "pre-commit",
            "pylint",
            "flake8",
            "black",
            "isort",
            "pytest",
            "pytest-cov",
            "sphinx",
            "sphinx-rtd-theme",
        ]
    },
    license="Apache License, Version 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
    ],
)
