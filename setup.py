from setuptools import setup, find_packages
import os

# For guidance on setuptools best practices visit
# https://packaging.python.org/guides/distributing-packages-using-setuptools/
project_name = os.getcwd().split("/")[-1]
version = "0.0.1"
package_description = "latent diffusion for forced monthly climate prediction"
url = "https://github.com/kjhall01/" + project_name

# Classifiers listed at https://pypi.org/classifiers/
classifiers = ["Programming Language :: Python :: 3"]
setup(name="md", # Change
      version=version,
      description=package_description,
      url=url,
      author="UMD Pareto Group",
      license="CC0 1.0",
      classifiers=classifiers,
      packages=find_packages(include=["md"]))
