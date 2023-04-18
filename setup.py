from setuptools import setup, find_packages

setup(
    name='SemanticML',
    version='0.1.1',
    description="CLI tool to perform semantic segmentation of 3D point clouds using Machine Learning algorithms.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author = 'Anass Yarroudh',
    author_email = 'ayarroudh@uliege.be',
    url = 'https://github.com/Yarroudh/Optim3D',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click==8.1.3',
        'numpy==1.24.2',
        'matplotlib==3.7.1',
        'scikit-learn==1.2.2',
        'laspy==2.4.1'
    ],
    entry_points='''
        [console_scripts]
        sml=SemanticML.main:cli
    '''
)
