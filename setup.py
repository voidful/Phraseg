from setuptools import setup, find_packages


setup(
    name='phraseg',
    version='1.0.2',
    description='unsupervised phrase discover - 無監督新詞發現',
    long_description="Github : https://github.com/voidful/phraseg",
    url='https://github.com/voidful/phraseg',
    author='Eric Lam',
    author_email='voidful.stack@gmail.com',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='phrase nlp word segment 斷詞 分詞 新詞發現 新詞',
    packages=find_packages()
)
