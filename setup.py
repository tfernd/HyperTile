from setuptools import setup, find_packages

setup(
    name="hyper_tile",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "diffusers",
        "einops",
    ],
    author="Thales Fernandes",
    author_email="thalesfdfernandes@gmail.com",
    description="Tiled-optimizations for Stable-Diffusion",
    url="https://github.com/tfernd/HyperTile",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)