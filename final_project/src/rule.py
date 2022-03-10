import re
from typing import List, Tuple, Dict
from pyparsing import nestedExpr
from collections import OrderedDict, defaultdict

from type import *
from variable import *


class Conditions:
    """
    Define fuzzy conditions. There should be at most two assertions
    """
    def __init__(self,
                 conditions: List = None,
                 op: Dict = None,
                 not_: List[bool] = False):
        self.conditions: List = conditions
        self.op: Dict = op
        self.not_: List[bool] = not_


class Conclusion:
    """
    Define fuzzy conclusion. There should be one assertion
    """
    def __init__(self,
                 conclusion: Tuple = None,
                 not_: bool = False):
        self.conclusion: Tuple = conclusion
        self.not_: bool = not_


class Rule:
    """
    Define rules parsing functions.
    """
    @staticmethod
    def parse_inp(rule: str) -> Tuple[List, Dict, List]:
        """
        Parse input string.
        Adapted from: https://github.com/Luferov/FuzzyLogicToolBox
        """
        clean_rule: str = ''
        for ch in rule:
            if ch in ['(', ')']:
                if not (len(clean_rule) > 0 and clean_rule[-1] == ' '):
                    clean_rule = f'{clean_rule} '
                clean_rule = f'{clean_rule}{ch} '
            else:
                if not (ch == ' ' and len(clean_rule) > 0 and clean_rule[-1] == ''):
                    clean_rule = f'{clean_rule}{ch}'
        clean_rule: str = re.sub(' +', ' ', clean_rule).strip()

        expressions = []
        words: List[str] = clean_rule.split(' ')
        for word in words:
            expressions.append(word)

        then_index: int = expressions.index('then')

        condition_expressions = expressions[1: then_index]
        conclusion_expression = expressions[then_index + 1:]

        then_index_conditions = rule.index('then')
        cut_rule = rule[3: then_index_conditions-1]
        sliced_rule = nestedExpr().parseString("(%s)" % cut_rule).asList()
        conditions_op_dict = OrderedDict()
        conditions_list = []

        layer, count, index = 5, 0, 0

        def divide_conditions(x, layer, count):
            if 'and' in x or 'or' in x:
                conditions_op_dict[count, layer] = x[1]
                # print(count, layer)
                count += 1
                divide_conditions(x[0], layer - 1, count)

                divide_conditions(x[2], layer + 1, count)
            else:
                conditions_list.append(x)

        divide_conditions(sliced_rule[0], layer, count)

        if len(conditions_op_dict) == 0:
            conditions_list = conditions_list[0]

        # return condition_expressions, conclusion_expression
        return conditions_list, conditions_op_dict, conclusion_expression

    @staticmethod
    def parse_conditions(conditions_list: List, conditions_op_dict: Dict) -> Conditions:
        """
        Parse condition string.
        """
        # operator: FuzzyRuleOperatorType = FuzzyRuleOperatorType.AND

        for i in conditions_op_dict:

            if 'and' == conditions_op_dict[i]:
                conditions_op_dict[i] = FuzzyRuleOperatorType.AND
            elif 'or' == conditions_op_dict[i]:
                conditions_op_dict[i] = FuzzyRuleOperatorType.OR
            else:
                operator = None

        not_ = []
        for i in conditions_list:
            if 'not' in i:
                not_.append(True)
            else:
                not_.append(False)

        # is_index: List = [i for i, x in enumerate(conditions_list) if x == 'is']
        # for i in is_index:
        #     if conditions_list[i + 1] == 'not':
        #         not_.append(True)
        #     else:
        #         not_.append(False)

        variables = []
        for i in conditions_list:
            variables.append(i[0])
        # left_bracket_index = [i for i, x in enumerate(condition_expressions) if x == '(']
        # for i in left_bracket_index:
        #     variables.append(condition_expressions[i + 1])

        domains = []
        for i in conditions_list:
            domains.append(i[-1])
        # right_bracket_index: List = [i for i, x in enumerate(condition_expressions) if x == ')']
        # for i in right_bracket_index:
        #     domains.append(condition_expressions[i - 1])

        # variables and domains should have same length
        assert len(variables) == len(domains), 'Parse Failed'
        assert len(conditions_op_dict) == len(variables) - 1, 'Parse Failed'

        conditions = []
        for i in range(len(variables)):
            conditions.append((variables[i], domains[i]))

        return Conditions(conditions, conditions_op_dict, not_)

    @staticmethod
    def parse_conclusion(conclusion_expressions: List) -> Conclusion:
        """
        Parse conclusion string.
        """
        not_: bool = True
        if 'not' in conclusion_expressions:
            not_ = True
        else:
            not_ = False

        variable: str = conclusion_expressions[1]
        domain: str = conclusion_expressions[-2]

        conclusion: tuple = (variable, domain)

        return Conclusion(conclusion, not_)

    def parse_rule(self, rule_str: str) -> (Conditions, Conclusion):
        """
        Parse rules.
        """
        conditions_list, conditions_op_dict, conclusion_expression = self.parse_inp(rule_str)

        conditions: Conditions = self.parse_conditions(conditions_list, conditions_op_dict)
        conclusion: Conclusion = self.parse_conclusion(conclusion_expression)

        return conditions, conclusion


class ParseHelper:
    keyword: List[str] = [
        'if',
        'then',
        'is',
        'not',
        'and',
        'or'
    ]


if __name__ == '__main__':
    test1 = 'if (input1 is mf1) and (input2 is mf1) then (output is mf1)'
    test2 = 'if (input1 is not mf2) or (input2 is mf1) then (output is mf1)'
    test3 = 'if (input1 is not mf2) then (output is mf1)'
