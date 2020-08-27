import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spacekit",
    version="0.0.1",
    author="Ru Keïn",
    author_email="hakkeray@gmail.com",
    description="Python package for Astronomical Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hakkeray/spacekit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
