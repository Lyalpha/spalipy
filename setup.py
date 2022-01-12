# spalipy - Detection-based astrononmical image registration
# Copyright (C) 2018-2021  Joe Lyman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spalipy",
    version="3.0.2",
    author="Joe Lyman",
    author_email="joedlyman@gmail.com",
    description="Detection-based astrononmical image registration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lyalpha/spalipy",
    download_url="https://github.com/Lyalpha/spalipy/archive/3.0.2.tar.gz",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "align-fits=spalipy.spalipy:main",
            "align-fits-simple=spalipy.spalipy:main_simple",
        ]
    },
    install_requires=["astropy>=3.2.2", "numpy>=1.10", "scipy>=1.1.0", "sep>=1.1.1"][::-1],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
