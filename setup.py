from setuptools import find_packages, setup

setup(
    name="oxebots-rl-ssl",
    version="1.0.0",
    description="OxeBots SSL and VSS robot soccer gym environments and RL training framework",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=1.0.0",
        "rc-robosim>=1.2.0",
        "pygame>=2.6.0",
        "protobuf>=3.20.0",
        "pyyaml>=6.0.0",
    ],
)
