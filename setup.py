# setup.py
from setuptools import find_packages, setup

setup(
    name="HUGIN_gym",  # Changed to match pyproject.toml (with underscore)
    version="0.1.0",
    description="OpenAI Gym environment for the HUGIN AUV",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "HUGIN_gym": ["assets/*"],  # Changed to match package name
    },
    install_requires=[
        "gymnasium>=0.29.1",
        "meshcat>=0.3.2",
        "numpy>=2.1.3",
        "scipy>=1.14.1",
        "stable-baselines3>=2.3.2",
        "rich",
        "tqdm",
        "tensorboard",
        "scikit-learn>=1.8.0", # modifiable
        "matplotlib",
    ],
)
