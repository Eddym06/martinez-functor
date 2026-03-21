from setuptools import setup, find_packages

setup(
    name="martinez-functor",
    version="0.1.0",
    description="Mathematical Core Engine for Universal Reduction Theorem",
    author="Eddy Martinez",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0"
    ],
    extras_require={
        "jax": ["jax>=0.4.0"]
    },
    python_requires=">=3.8",
)
