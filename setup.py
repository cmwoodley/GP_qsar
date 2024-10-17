from setuptools import setup, find_packages

setup(
    name='GP_qsar',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'rdkit',
        'scikit-learn',
        "torch",
        'gpytorch',
        "optuna",
    ],
    description='GP regressor for use in reinvent and modAL',
    author='Chris',
)
