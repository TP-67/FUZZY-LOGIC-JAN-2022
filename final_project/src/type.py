"""
Conduct set operations among two fuzzy sets.

Definition:
    Define set operations for linguistic terms.
"""

from enum import Enum


class CompositionType(Enum):
    """
    Unused
    """
    MIN = 1
    MAX = 2
    PROD = 3
    SUM = 4


class AndMethod(Enum):
    """
    Different choices of 'and' operation when using FuzzyRuleOperatorType.AND
    """
    # min(x, y)
    MIN = 1

    # prod(x, y)
    PROD = 2


class OrMethod(Enum):
    """
    Different choices of 'or' operation when using FuzzyRuleOperatorType.OR
    """
    # max(x, y)
    MAX = 1

    # add(x, y) - prod(x, y)
    PROB = 2


class FuzzyRuleOperatorType(Enum):
    """
    Operator between different assertions, depending on the input rules. One operator for each rule.
    """
    AND = 1
    OR = 2


class InferenceMethod(Enum):
    """
    Inference methods when inference conclusion in a rule based on conditions.
    """
    MIN = 1
    PROD = 2


class AggregationMethod(Enum):
    """
    Aggregation method when aggregate multiple rules.
    """
    MAX = 1
    SUM = 2


class DefuzzificationMethod(Enum):
    """
    Different defuzzification methods including centroid and average maximum methods taught on lectures.
    """
    CENTROID = 1
    AVERAGE_MAXIMUM = 2
