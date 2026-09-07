"""One-command action-independent observation training; keeps legacy runs separate."""
import sys
from train_hokoff_decisions import main

if __name__=='__main__':
    main(['--sampling','independent','--delay-short-weight','128','--timing-positive-weight','8']+sys.argv[1:])
