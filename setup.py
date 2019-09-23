from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="AMP-proc",
    version="0.1",
    scripts="[point_triangulation.py]",
    author="Mitchell Scott",
    author_email="miscott@uw.edu",
    description="An AMP image processing package",
    long_description=long_description,
)
