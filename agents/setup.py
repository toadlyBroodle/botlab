from setuptools import setup, find_packages

setup(
    name="agents",
    version="0.1.0",
    description="Small, efficient AI agents for automation",
    author="Landon Mutch",
    author_email="landonmutch@protonmail.com",
    packages=find_packages(),
    python_requires=">=3.11,<3.14",
    include_package_data=True,
) 