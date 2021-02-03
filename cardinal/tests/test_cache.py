import os
from cardinal import cache
from tempfile import NamedTemporaryFile


def _test_backend(backend_class, suffix=''):
    f = NamedTemporaryFile()
    filename = f.name + suffix
    f.close()
    be = backend_class(filename)
    assert(len(be.get('test')) == 0)
    be._store('test', 37, key='example')
    be.close()
    be = backend_class(filename)
    assert be.get('test').to_dict() == {'key': {0: 'example'}, 'value': {0: 37}}
    be.close()
    os.unlink(filename)


def test_backends():
    _test_backend(cache.ShelveStore, '.db')
    _test_backend(cache.SqliteStore)
