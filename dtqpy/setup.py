from setuptools import setup
from setuptools import find_packages


__version__ = "0.0.1"

setup(name='dtqpy',
    version=__version__,
    description='dtqpy',
    license='BSD-3',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'mat4py',
        'osqp',
    ],
    zip_safe=False,
)