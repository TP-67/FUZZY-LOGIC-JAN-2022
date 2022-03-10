from typing import List, Tuple
from domain import Domain


class FuzzyVariable:

    def __init__(self, name: str, min_vale: float = 0.0, max_value: float = 0.0, *domain: Domain):
        assert min_vale < max_value, f"{min_vale} >? {max_value}"

        self.name: str = name
        self.min_value: float = min_vale
        self.max_value: float = max_value
        self.domain: List[Domain] = list(domain)

    def get_domain_by_name(self, name: str) -> Domain or None:
        for d in self.domain:
            if d.name == name:
                return d

    @property
    def value(self):
        return self.domain
