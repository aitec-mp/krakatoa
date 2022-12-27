from setuptools import setup

setup(
  name = 'krakatoa',       
  packages = ['krakatoa', 'krakatoa/models', 'krakatoa/future'],  
  version = '0.0.5',      
  license='MIT',        
  description = 'Machine Learning high level package.',  
  author = 'Matheus de Prá Andrade',              
  author_email = 'mpandrade@ucs.br',    
  url = 'https://github.com/aitec-mp/krakatoa',  
  download_url = 'https://github.com/aitec-mp/krakatoa/archive/refs/tags/0.0.5.tar.gz',    
  keywords = ['krakatoa', 'machine learning'],  
  install_requires=[    
          'sklearn>=1.1',
          'numpy>=1.20',
          'pandas>=1.0.0',
          'xgboost>=1.5.0'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',     
    'Intended Audience :: Developers',  
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
