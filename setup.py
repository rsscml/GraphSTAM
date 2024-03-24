
from pathlib import Path
from setuptools import setup

# The directory containing this file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="GraphSTAM",
    version="1.2.10",
    description="Graph Based Spatio-Temporal Attention Models For Demand Forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rahul Sinha",
    author_email="rahul.sinha@unilever.com",
    packages=["graphstam", "BasicGraph", "SpatialTemporalGraph", "TemporalSpatialGraph", "optimized.graphstam",
              "optimized.BasicGraph", "optimized.SpatialTemporalGraph", "optimized.TemporalSpatialGraph",
              "probabilistic.graphstam", "probabilistic.BasicGraph", "subgraphsampling.graphstam", "subgraphsampling.BasicGraph",
              "smallgraph.graphstam", "smallgraph.BasicGraph", "hierarchical.graphstam", "hierarchical.BasicGraph",
              "simplified.graphstam", "simplified.BasicGraph", "simplified.HierarchicalGraph",
              "simplified.MultistepHierarchicalGraph", "simplified.SmallGraph"],
    include_package_data=True,
    install_requires=['statsmodels', 'scipy', 'tweedie']
)