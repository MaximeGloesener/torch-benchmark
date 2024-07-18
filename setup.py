from setuptools import setup, find_packages

setup(
    name="pytorch_bench",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "torch",
        "pynvml",
        "matplotlib",
        "numpy",
        "colorama",
        "torchprofile"
    ],
    author="Maxime Gloesener",
    author_email="max.gleu@gmail.com",
    description="torch benchmarking tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MaximeGloesener/torch_benchmark",
)