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
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ),
)
