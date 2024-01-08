from setuptools import setup, find_packages

setup(
      name = 'dfsm',
      version = '1.0.0',
      description = 'Module to construct derivative function surrogate models (DFSM)',
      author = ['Athul Krishna Sundarrajan', 'Daniel R. Herber'],
      author_email = ['Athul.Sundarrajan@colostate.edu','daniel.herber@colostate.edu'],
      packages = find_packages(),
      install_requires = ['numpy','scipy','matplotlib','sklearn'],
      license = "Apache License, Version 2.0"
      )

