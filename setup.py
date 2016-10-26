import os
from setuptools import setup, find_packages

def read(fname):
  """
  Utility function to read specified file.
  """
  path = os.path.join(os.path.dirname(__file__), fname)
  return open(path).read()

setup(name="bcikit",
      version="0.0.2",
      description="bcikit decoupled from cloudbrain",
      packages=find_packages(),
      install_requires=[
          'mne',
          'numpy',
          'pika',
          'pyliblo',
          'pyriemann',
          'pyserial',
          'scipy',
          'simplejson',
          'scikit-learn',
          'pyyaml'
      ],
      include_package_data=True,
      long_description=read("README.md"),
      license='GNU Affero General Public License v3',
      classifiers=[
          'License :: OSI Approved :: GNU Affero General Public License v3'
      ],
      entry_points = {}
      )
