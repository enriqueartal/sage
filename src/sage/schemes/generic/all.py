# code exports
from sage.misc.lazy_import import lazy_import

from .spec import Spec
from .hypersurface import ProjectiveHypersurface, AffineHypersurface

lazy_import('sage.schemes.generic.zeta_function', 'ZetaFunctions')
