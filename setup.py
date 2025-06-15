from setuptools import setup, find_packages

setup(
    name='genesis-module',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'psutil>=5.0',
    ],
    extras_require={
        'tests': ['pytest'],
    },
    author='Manus AI',
    description='A PyTorch module for human-like learning with self-replay, ethical gating, and memory efficiency.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    license_files=['LICENSE'],
    url='https://github.com/manus-ai/genesis-module', # Placeholder URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
)


