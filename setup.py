from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='tcr-pmhc binding',
   version='0.1.0',
   description='TCR-pMHC binding prediction',
   long_description=long_description,
   license="MIT",
   author='Jean-Guillaume Brasier',
   author_email='jbrasier@g.harvard.edu',
   packages=find_packages(include=['src']),#same as name
   package_dir={'src': 'src/'}
#    install_requires=[
#                     'numpy',
#                     'matplotlib',
#                     'scikit-learn==1.1.2',
#                     'torch==1.12.1',
#                     'torch-cluster==1.6.0',
#                     'torch-geometric==2.1.0.post1',
#                     'torch-scatter==2.0.9',
#                     'torch-sparse==0.6.15'
#                     ], #external packages as dependencies
)