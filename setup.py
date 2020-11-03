import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spalipy",
    version="2.0.3",
    author="Joe Lyman",
    author_email="joedlyman@gmail.com",
    description="Detection-based astrononmical image registration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lyalpha/spalipy",
    download_url="https://github.com/Lyalpha/spalipy/archive/2.0.3.tar.gz",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "align-fits=spalipy.spalipy:main",
            "align-fits-simple=spalipy.spalipy:main_simple",
        ]
    },
    install_requires=["numpy>=1.10", "astropy>=3.2.2", "scipy>=1.1.0", "sep~=1.0.3"][::-1],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
