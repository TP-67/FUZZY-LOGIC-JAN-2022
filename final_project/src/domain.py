from mf import *


class Domain:
    """
    Define fuzzy domain.
    """
    def __init__(self, name: str, mf: MembershipFunction):
        self.name: str = name
        self.mf: MembershipFunction = mf
