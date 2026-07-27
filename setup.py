from setuptools import setup

setup(
    name="privacuguard-fl",
    version="0.1.0",
    packages=[
        "src",
        "src.attacks",
        "src.core",
        "src.data",
        "src.deployment",
        "src.differential_privacy",
        "src.ui",
    ],
)
