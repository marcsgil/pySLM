import setuptools
from pkg_resources import parse_requirements
import m2r2
import sphinx.cmd.build
import sphinx.ext.apidoc
import pathlib
import shutil
import webbrowser

from optics import log, __version__

log.info('Reading the package requirements...')
# Require that all packages in requirements.txt are installed prior to this
with open('requirements.txt') as file:
    requirements = [str(req) for req in parse_requirements(file)]


log.info('Building the documentation...')
log.info('Converting README.md to rst...')
long_description_rst = m2r2.parse_from_file('README.md')

code_path = pathlib.Path(__file__).parent.resolve()
docs_path = code_path / 'docs'
apidoc_path = docs_path / 'source/api'  # a temporary directory
html_output_path = docs_path / 'build/html'
log.info(f'Removing temporary directory {apidoc_path}...')
shutil.rmtree(apidoc_path, ignore_errors=True)
sphinx.ext.apidoc.main(['-f', '-d', '4', '-M',
                        '-o', f"{apidoc_path}",
                        f"{code_path / 'optics'}",  # include this
                        f"{code_path / 'optics/experimental'}",  # exclude this and the following
                        f"{code_path / 'optics/external'}"
                        ]
                       )
log.info(f'Removing old documentation in {html_output_path}...')
shutil.rmtree(html_output_path, ignore_errors=True)
log.info('Building html...')
ret_value = sphinx.cmd.build.main(['-M', 'html', f"{docs_path / 'source'}", f"{docs_path / 'build'}"])
if ret_value != 0:
    log.error(f'sphinx-build returned {ret_value}.')
log.info(f'Removing temporary directory {apidoc_path}...')
shutil.rmtree(apidoc_path, ignore_errors=True)

build_path = docs_path / 'build/html/index.html'
log.info(f'Opening documentation at {build_path}...')
webbrowser.open(str(build_path))

setuptools.setup(
    name='optics',
    version=__version__,
    keywords='optical microscopy',
    packages=['optics'],  #find_packages(),
    include_package_data=True,
    author='Tom Vettenburg',
    author_email='t.vettenburg@dundee.ac.uk',
    description=('Library for controlling optics and microscopes in the lab.'),
    long_description=long_description_rst,
    #long_description_content_type='text/markdown',
    long_description_content_type='text/x-rst',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
)
