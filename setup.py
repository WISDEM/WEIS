import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RAFT",
    version="0.1",
    author="National Renewable Energy Laboratory",
    author_email="matthew.hall@nrel.gov",
    description="RAFT: Response Amplitudes of Floating Turbines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WISDEM/RAFT",
    packages=['raft'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
