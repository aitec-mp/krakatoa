
from distutils.core import setup
setup(
  name = 'krakatoa',       
  packages = ['krakatoa'],  
  version = '0.0.3',      
  license='MIT',        
  description = 'Machine Learning high level package.',  
  author = 'Matheus de Prá Andrade',              
  author_email = 'mpandrade@ucs.br',    
  url = 'https://github.com/aitec-mp/krakatoa',  
  download_url = 'https://github.com/aitec-mp/krakatoa/archive/refs/tags/0.0.3.tar.gz',    
  keywords = ['krakatoa', 'machine learning'],  
  install_requires=[    
          'sklearn',
          'numpy',
          'pandas',
          'xgboost'
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
