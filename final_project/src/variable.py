from typing import List, Tuple, Dict
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


class SugenoFuzzyFunction:

    def __init__(self, name: str, coefficient: Dict[FuzzyVariable, float], const: float):
        self.name: str = name
        self.coefficient: Dict[FuzzyVariable, float] = coefficient
        self.const = const


class SugenoFuzzyVariable:

    def __init__(self, name: str, *sugenofunction: SugenoFuzzyFunction):
        self.name: str = name
        self.sugenofunction: List[SugenoFuzzyFunction] = list(sugenofunction)

    def get_function_by_name(self, name: str) -> SugenoFuzzyFunction or None:
        for s in self.sugenofunction:
            if s.name == name:
                return s

    @property
    def value(self):
        return self.sugenofunction
