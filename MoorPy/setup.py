import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MoorPy",
    version="0.9.0",
    author="National Renewable Energy Laboratory",
    author_email="matthew.hall@nrel.gov",
    description="A design-oriented mooring system library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NREL/MoorPy",
    packages=['moorpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: LGPL",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml"
    ],
)
