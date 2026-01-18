from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("-e")
        ]

setup(
    name="ML-PROJECT",
    version="0.0.1",
    author="Swagat Mishra",
    author_email="mishraswagat2804@gmail.com",
    packages=find_packages(),
    install_requires=read_requirements(),
)
