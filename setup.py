from setuptools import find_packages, setup

setup(
    name="shipment",       
    version="0.0.1",                       
    author="Suraj Yadav",                    
    author_email="suraj.yadav.aiml@gmail.com", 
    packages=find_packages(),                 # Automatically find project packages
    install_requires=[],                    

    description="A package for predicting shipment prices", 
    long_description=open("README.md").read(), 
    url="https://github.com/suraj-yadav-aiml/Shipment-Price-Prediction-End-to-End-Project",  
    python_requires='>=3.8',                 
)
