from setuptools import setup, find_packages

setup(
    name='GP_qsar',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'rdkit',
        'scikit-learn',
        # Add any other dependencies your project requires
    ],
    description='GP regressor for use in reinvent and modAL',
    author='Chris',
)
