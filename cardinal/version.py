# *- encoding: utf-8 -*-
"""
cardinal version, required package versions, and utilities for checking
"""
# Author: Alexandre Abraham

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
# X.Y
# X.Y.Z # For bugfix releases
#
# Admissible pre-release markers:
# X.YaN # Alpha release
# X.YbN # Beta release
# X.YrcN # Release Candidate
# X.Y # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.0.7'

# This is a tuple to preserve order, so that dependencies are checked
# in some meaningful order (more => less 'core').
DEPENDENCIES_METADATA = (
    ('numpy', {'min_version': '1.11'}),
    ('scipy', {'min_version': '0.19'}),
    ('scikit-learn', {
        'min_version': '0.23',
        'extra_options': ['sklearn', 'examples', 'submodular', 'doc']}),
    ('apricot-select', {
        'min_version': '0.5.0', 'extra_options': ['submodular', 'doc']}),
    ('matplotlib', {
        'min_version': '2.0', 'extra_options': ['examples', 'doc']}),
    ('sphinx-gallery', {
        'min_version': '0.5.0', 'extra_options': ['doc']}),
    ('sphinx', {
        'min_version': '2.2.2', 'extra_options': ['doc']}),
)

package_to_module = {
    'scikit-learn': 'sklearn',
    'apricot-select': 'apricot'
}


def check_modules(extra_option=None, import_module=None, strict=True):
    """Check that module is installed with a recent enough version

    Args:
        extra_option: If None, check based modules, otherwise checks
            modules for the specified option. None by default.
        import_module: If the check is made in a specific module, adds
            it to error messages. Empty by default.
        strict: If True (default), raises an error in case of problem.
            Otherwise returns a boolean indicating if the set up is ok.
    """
    from distutils.version import LooseVersion

    if import_module:
        import_module = '.' + import_module

    for package_name, metadata in DEPENDENCIES_METADATA:

        if not ((extra_option is None and 'extra_options' not in metadata)
                or (extra_option in metadata.get('extra_options', []))):
            continue

        min_version = metadata['min_version']
        try:
            module_name = package_to_module.get(package_name, package_name)
            module = __import__(module_name)
        except ImportError as exc:
            user_friendly_info = (
                'Module "{0}" could not be found. '
                'Please install it properly to use cardinal{1}.'.format(
                    package_name, import_module))
            exc.args += (user_friendly_info,)
            exc.msg += '. ' + user_friendly_info
            if strict:
                raise
            else:
                return False

        # Avoid choking on modules with no __version__ attribute
        module_version = getattr(module, '__version__', '0.0.0')

        version_too_old = (not LooseVersion(module_version) >=
                           LooseVersion(min_version))

        if version_too_old:
            message = (
                'A {package_name} version of at least {minimum_version} '
                'is required to use cardinal{import_module}. '
                '{module_version} was found. '
                'Please upgrade {package_name}').format(
                    package_name=package_name,
                    minimum_version=min_version,
                    module_version=module_version,
                    import_module=import_module)

            if strict:
                raise ImportError(message)
            else:
                return False
    if not strict:
        return True
