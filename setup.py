from setuptools import find_packages, setup

setup(
    name="privacuguard-fl",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
)
