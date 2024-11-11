from setuptools import setup, find_packages

setup(
   name='simulation',
   version='1.0',
   description='Diffusion simulation and Monte Carlo',
   author='MrG1raffe',
   author_email='dimitri.sotnikov@gmail.com',
   packages=find_packages(),
   install_requires=['numpy', 'typing'], #external packages as dependencies
)