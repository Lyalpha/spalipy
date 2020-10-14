import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spalipy",
    version="1.0",
    author="jdl",
    author_email="joedlyman@gmail.com",
    description="Detection-based astrononmical image registration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lyalpha/spalipy",
    download_url="https://github.com/Lyalpha/pympc/archive/v1.0.tar.gz",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.10",
        "astropy>=3.1.0",
        "scipy>=1.1.0",
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ),
)
