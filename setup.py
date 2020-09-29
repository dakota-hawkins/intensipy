from setuptools import setup
# long_description = """
# # intensipy
# Normalize intensity values in 3D image stacks.

# Python implementation of the Intensify3D algorithm originally developed by [Yoyan et al](https://www.nature.com/articles/s41598-018-22489-1).
# """

with open("README.md", "r") as f:
    long_description = f.read()

setup(name='intensipy',
      version='0.1.0',
      description="Normalize intensity values in 3D images.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/dakota-hawkins/intensipy',
      author='Dakota Y. Hawkins',
      author_email='dyh0110@bu.edu',
      license='BSD',
      packages=['intensipy'],
      install_requires=['numpy',
                        'scipy',
                        'scikit-image',
                        'statsmodels'],
      classifiers=["Programming Language :: Python :: 3"])