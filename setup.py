from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(name='ml_ops',
      version='0.0.1',
      author="Vol1and",
      packages=find_packages(),
      install_requires=requirements
      )