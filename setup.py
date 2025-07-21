from setuptools import setup, find_packages

setup(
    name="nb_sim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click",
        "numpy",
        "torch",
        "scipy",
        "biopython",
        "matplotlib"
    ],
    python_requires=">=3.8",
    description="Nonlinear Rigid Block Normal-Mode Analysis Simulator",
    author="Sergey Martynenko",
    license="MIT",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nb_sim = nb_sim.__main__:main"
        ]
    }
)
