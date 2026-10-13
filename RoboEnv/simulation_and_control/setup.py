from setuptools import setup, find_packages

setup(
    name='simulation_and_control',  # Updated to match your GitHub repo name
    version='0.1',
    author='VModugno',
    author_email='valerio.modugno@gmail.com',
    description='A simple package to provide an interface to the PyBullet simulator and Pinocchio for robotic simulation and control.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/VModugno/simulation_and_control',  # Updated to HTTPS and correct URL
    packages=find_packages(),
    #install_requires=[
    #    'pybullet',  # No version specified
    #    'pinocchio'  # No version specified
    #],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',  # Added since you mentioned teaching
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)

