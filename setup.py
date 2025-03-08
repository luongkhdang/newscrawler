from setuptools import setup, find_packages

setup(
    name="newscrawler",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "newspaper4k>=0.1.0",
        "requests>=2.28.1",
        "python-dateutil>=2.8.2",
        "beautifulsoup4>=4.11.1",
        "lxml>=4.9.1",
        "urllib3>=1.26.12",
        "reppy>=0.4.14",
        "pydantic>=2.0.0",
        "fastapi>=0.95.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.5",
        "langchain>=0.0.267",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "mypy>=1.3.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.10",
) 