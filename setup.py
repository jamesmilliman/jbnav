from setuptools import setup, find_packages

setup(
    name="jbnav",
    version="0.1",
    packages=find_packages(),
    license="MIT",
    description="TODO",
    long_description=open("README.md").read(),
    install_requires=["numpy", "jsonlines", "opencv-python", "tqdm"],
    author="James Milliman",
    author_email="jaimsmilliman@gmail.com",
)
