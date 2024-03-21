import os
import gzip, pickle

def dump(fname, obj):
    if not fname.endswith('.pkl.gz'):
        fname += '.pkl.gz'
    with gzip.open(fname, 'wb') as f_out:
        pickle.dump(obj, f_out)
    print('- dumped', fname)

def load(fname):
    if not fname.endswith('.pkl.gz'):
        fname += '.pkl.gz'
    with gzip.open(fname, 'rb') as f:
        mus = pickle.load(f)
    print('- loaded', fname)
    return mus

def runF(fname, overwrite, func, *args, **kwargs):
    """
    before running a function it checks if we have already something saved
    """
    path = os.path.join('/tmp', fname)
    if not path.endswith('.pkl.gz'):
        path += '.pkl.gz'
    if not os.path.exists(path) or overwrite:
        ret = func(*args, **kwargs)
        dump(path, ret)
    else:
        ret = load(path)
    return ret
