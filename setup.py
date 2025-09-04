from setuptools import setup, find_packages

setup(
    name='pyFMST',
    version='0.1.1',
    description='Python translation layer for basic FMST functionality',
    author='Will Pizii',
    packages=find_packages(),
    install_requires=[
        # List any dependencies here, e.g.,
        'numpy>=1.18',
        'pandas>=1.1',
        'obspy',
        'pygmt',
        'matplotlib',
	'tqdm'
    ],
    project_urls={
        'Source': 'https://github.com/willpizii/pyfmst'
    },
)
