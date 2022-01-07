import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# taken from https://github.com/CEA-COSMIC/ModOpt/blob/master/setup.py
with open('requirements.txt') as open_file:
    install_requires = open_file.read()

setuptools.setup(
    name="zac-ramzi-sub-scripts",
    version="0.0.1",
    author="Zaccharie Ramzi",
    author_email="zaccharie.ramzi@gmail.com",
    description="Submission scripts for JZ and TGCC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zaccharieramzi/submission-scripts",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    scripts=['jean_zay/hydra_config/bin/submitit-hydra-launch', 'jean_zay/sync_wandb.sh', 'jean_zay/syncall_wandb.sh'],
    install_requires=install_requires,
    python_requires='>=3.6',
    include_package_data=True,
)
