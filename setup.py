from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='diffcse',
    version='0.1.0',
    description='A sentence embedding tool based on DiffCSE',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yung-Sung Chuang et al.',
    url='https://github.com/voidism/DiffCSE',
    download_url='https://github.com/voidism/DiffCSE/archive/refs/tags/v0.1.0.tar.gz',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        "scipy==1.5.4",
        "datasets==1.2.1",
        "pandas==1.1.5",
        "scikit-learn==0.24.0",
        "prettytable==2.1.0",
        "gradio",
        "torch==1.7.1",
        "setuptools==49.3.0",
    ]
)
