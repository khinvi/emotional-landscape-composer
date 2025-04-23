from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="emotional-landscape-composer",
    version="0.1.0",
    description="Transform landscapes into music using AI and geography!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/khinvi/emotional-landscape-composer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: MIDI",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Sound/Audio :: Composers",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'landscape-composer=compose_music:main',
        ],
    },
)