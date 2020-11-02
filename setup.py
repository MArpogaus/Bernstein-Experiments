from setuptools import setup, find_packages

setup(
    name="bernstein_paper",
    packages=find_packages('src'),
    package_dir={"": "src"},
    install_requires=[
        'scikit-learn',
        'tensorflow',
        'tensorflow_probability',
        'matplotlib',
        'seaborn',
        'numpy'
    ]
)
