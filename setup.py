from setuptools import setup

setup(
    name="jbnav",
    version="0.1",
    py_modules=["jbnav.py"],
    license="MIT",
    description="TODO",
    long_description=open("README.md").read(),
    install_requires=["numpy", "jsonlines", "opencv-python"],
    author="James Milliman",
    author_email="jaimsmilliman@gmail.com",
)
