import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from .affine_registration import affine_registration  # noqa
from .rigid_registration import rigid_registration  # noqa
from .deformable_registration import deformable_registration  # noqa
from .jrmpc_rigid import jrmpc_rigid  # noqa
