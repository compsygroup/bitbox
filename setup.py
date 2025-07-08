from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name = 'bitbox',
    version = '2025.07dev',
    description = 'Behavioral Imaging Toolbox',
    author = 'ComPsy Group',
    author_email = 'tuncb@chop.edu',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/compsygroup/bitbox',
    classifiers = [
        'License :: CC-BY-NC-4.0',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires = '>=3.8',
    install_requires = [
        'scipy',
        'cvxpy',
        'numpy',
        'scikit-learn',
        'python-dateutil',
        'PyWavelets',
        'matplotlib',
        'pandas',
        'requests'
    ]
)
