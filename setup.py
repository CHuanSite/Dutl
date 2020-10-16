import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Dutl-hchen", # Replace with your own username
    version="0.0.1",
    author="Huan Chen",
    author_email="hchen130@jhu.edu",
    description="A python package to do deep unsupervised transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CHuanSite/Deep-Unsupervised-Transfer-Learning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
