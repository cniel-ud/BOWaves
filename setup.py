from setuptools import setup, find_packages
# import codecs
# import os
#
# here = os.path.abspath(os.path.dirname(__file__))
#
# with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
#     long_description = "\n" + fh.read()

VERSION = '0.0.26'
DESCRIPTION = 'Learning representative waveforms'
LONG_DESCRIPTION = 'Learning representative waveforms for time series clustering and dictionary learning'

# Setting up
setup(
    name="BOWaves",
    version=VERSION,
    author="ameek2 (CNIEL @ UD)",
    author_email="<austin.meek10@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    #install_requires=['scipy', 'scikit-learn', 'numpy'],
    keywords=['python', 'EEG', 'time series', 'dictionary learning'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)