import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="fastim",
    version="0.0.2",
    author="Bikram Saha",
    author_email="imbikramsaha@gmail.com",
    description="fastim helps you to import all the necessary ml and dl libraries by writing 1 line Code.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imbikramsaha/fastim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'seaborn',
        'fastai',
        'pathlib',
        'tqdm',
        'scipy',
        'fastbook',
        'timm',
        'transformers',
        'datasets',
        'kaggle',
        'defaults',
        'fastcore',
        'torchvision',
        'pandas',
        'requests',
        'pyyaml',
        'fastprogress',
        'pillow',
        'scikit-learn',
        'spacy',
        'pytorch',
    ],
)
