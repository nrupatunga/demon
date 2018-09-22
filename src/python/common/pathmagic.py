import os
import sys

examples_dir = os.path.dirname(__file__)
print(examples_dir)
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))
sys.path.insert(0, '../../demon/examples/pycpd/pycpd/')
sys.path.insert(0, '../../demon/lmbspecialops/python/')
