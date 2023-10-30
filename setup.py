
from pathlib import Path
from setuptools import setup

# The directory containing this file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="GraphSTAM",
    version="1.2",
    description="Graph Based Spatio-Temporal Attention Models For Demand Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rahul Sinha",
    author_email="rahul.sinha@unilever.com",
    packages=["graphstam", "BasicGraph", "SpatialTemporalGraph", "TemporalSpatialGraph"],
    include_package_data=True,
    install_requires=[]
)