
"""
Module to expose more detailed version info for the installed `numpy`
"""
version = "2.4.5"
__version__ = version
full_version = version

git_revision = "26e81854b03fea36ac2b21e3158756fdd40f19bb"
release = 'dev' not in version and '+' not in version
short_version = version.split("+")[0]
