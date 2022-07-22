
from distutils.core import setup
setup(
  name = 'krakatoa',       
  packages = ['krakatoa'],  
  version = '0.0.1',      
  license='MIT',        
  description = 'Machine Learning high level package.',  
  author = 'Matheus de Prá Andrade',              
  author_email = 'mpandrade@ucs.br',    
  url = 'https://github.com/aitec-mp/krakatoa',  
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['krakatoa', 'machine learning'],  
  install_requires=[            # I get to this in a second
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
