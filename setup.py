# import codecs
# import os
from setuptools import setup, find_packages

# # these things are needed for the README.md show on pypi
# here = os.path.abspath(os.path.dirname(__file__))
#
# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = '1.0.2'
DESCRIPTION = 'QCS: Quantum Correlation Solver'
LONG_DESCRIPTION = 'QCS is an open-source Python code that allows to study the single-photon transmission and reflection, ' \
                   'as well as the nth-order equal-time correlation functions (ETCFs) in driven-dissipative quantum systems.'
REQUIRES = ['numpy (>=1.8)', 'scipy (>=0.15)']
INSTALL_REQUIRES = ['numpy>=1.8', 'scipy>=0.15']
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
KEYWORDS = "quantum physics higher-order equal-time correlation function"

# Setting up
setup(
    name="qcs_phy",
    version=VERSION,
    author="ZhiGuang Lu",
    author_email="youngqlzg@gamil.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    keywords=KEYWORDS,
    requires=REQUIRES,
    platforms=PLATFORMS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    license="BSD",
    url="",
)
