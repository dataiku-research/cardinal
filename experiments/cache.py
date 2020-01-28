from contextlib import contextmanager


gcache = dict()

class Cached(Exception): pass

@contextmanager
def cache_key(key):

    @contextmanager
    def cache():
        if key in gcache:
            raise Cached()
        else:
            yield
    try:
        yield cache
    except Cached:
        print('Skipping flag')


with cache_key('one') as check, check():
    print('one')

with cache_key('two') as check, check():
    print('two')

with cache_key('five') as check, check():
    print('five')