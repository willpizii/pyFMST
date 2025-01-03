from setuptools import setup

setup(
    name='pyFMST',
    version='0.0.1',
    description='Python translation layer for basic FMST functionality',
    author='Will Pizii',
    packages=['pyFMST'],                      # root folder of your package
    install_requires=[
        # List any dependencies here, e.g.,
        'numpy>=1.18',
        'pandas>=1.1',
        'obspy',
        'pygmt',
        'matplotlib'
    ],
    project_urls={
        'Source': 'https://github.com/willpizii/pyfmst'
    },
)
