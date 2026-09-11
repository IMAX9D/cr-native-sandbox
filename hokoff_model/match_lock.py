"""Per-user endpoint lease: never let a second box reset an active match."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile


@contextmanager
def match_lease(host,port,*,directory=None):
    name=hashlib.sha256(f'{host}:{port}'.encode()).hexdigest()[:24]
    path=Path(directory or tempfile.gettempdir())/('cr-hokoff-match-'+name+'.lock')
    stream=path.open('a+b');locked=False
    try:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as error:
            raise RuntimeError('A match box already controls this host/port; close it before opening another.') from error
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()
