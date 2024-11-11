from setuptools import setup

setup(
   name='simulation',
   version='1.0',
   description='A useful module',
   author='MrG1raffe',
   author_email='dimitri.sotnikov@gmail.com',
   py_modules=['brownian_motion', 'diffusion', 'utility', 'monte_carlo'],
   install_requires=['numpy', 'typing'], #external packages as dependencies
)