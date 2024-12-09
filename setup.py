from setuptools import setup, find_packages

def gather_files():
    # List additional files needed for the package, such as main.py and files in fmstUtils directory
    package_files = ['main.py']
    fmst_utils_files = ['fmstUtils/fmstUtils.py', 'fmstUtils/genUtils.py']
    package_files.extend(fmst_utils_files)
    return package_files

setup(
    name='pyFMST',
    version='0.0.1',
    description='Python translation layer for basic FMST functionality',
    author='Will Pizii',
    packages=find_packages(),  # Automatically find all packages
    package_data={
        '': gather_files(),  # Include files in the root directory and fmstUtils subdirectory
    },
    install_requires=[
        # List any dependencies here, e.g.,
        'numpy>=1.18',
        'pandas>=1.1',
        'obspy',
        'pygmt',
        'matplotlib'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3 License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    project_urls={
        'Source': 'https://github.com/willpizii/pyfmst'
    },
)
