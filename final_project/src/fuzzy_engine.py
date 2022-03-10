from typing import List, Tuple, Dict
from collections import OrderedDict, defaultdict
import numpy as np

from rule import *
from variable import *
from type import *
from mf import *
from utils import *


class FuzzyEngine:
    def __init__(self,
                 inp: List[FuzzyVariable],
                 out: FuzzyVariable,
                 operator: FuzzyRuleOperatorType,
                 out_min: int,
                 out_max: int
                 ):
        self.inp: List[FuzzyVariable] = inp
        self.out: FuzzyVariable = out
        self.operator: FuzzyRuleOperatorType = operator
        # Range of the output fuzzy variable.
        self.out_min = out_min
        self.out_max = out_max
        assert out_min < out_max, f"{out_min} >? {out_max}"
        self.resolution = int(out_max - out_min) * 5

        # Append multiple rules together for fuzzy inference process.
        self.rules: List[Conditions, Conclusion] = []

        # Define different operator choices (we use default settings taught on lectures).
        self.and_method = AndMethod.MIN
        self.or_method = OrMethod.MAX
        self.inference_method = InferenceMethod.MIN
        self.aggregation_method = AggregationMethod.MAX

    def calculate(self, inp: Dict[FuzzyVariable, float]) -> Tuple:
        """
        Main function for calculating a final fuzzy result.
        (1) Fuzzify the variables based on input values.
        (2) Evaluate fuzzy conditions based on assertions in conditions.
        (3) Inference conclusion of each rule.
        (4) Aggregate multiple conclusions together to obtain the final fuzzy result.
        """
        fuzzy_matrix: Dict[FuzzyVariable, Dict[Domain, float]] = self.fuzzify(inp)
        conditions: List[float] = self.rule_evaluation(fuzzy_matrix)
        # (N, L)
        conclusions: List = self.inference(conditions)
        # (L,)
        fuzzy_result: List[float] = self.aggregation(conclusions)

        # # Check fuzzy matrix
        # import pprint
        # pprint.pprint(fuzzy_matrix)

        return fuzzy_matrix, conditions, conclusions, fuzzy_result

    def fuzzify(self, inp: Dict[FuzzyVariable, float]) -> Dict[FuzzyVariable, Dict[Domain, float]]:
        """
        Calculate a fuzzy matrix (ordered dictionary).
        """
        result = OrderedDict()
        for variable in self.inp:
            result[variable] = OrderedDict()
            for d in variable.domain:
                # Calculate fuzzy values based on every membership function for each input fuzzy variables.
                result[variable][d] = d.mf.get_value(inp[variable])
        return result

    # def fuzzify(self, inp: Dict[FuzzyVariable, float]) -> Dict[FuzzyVariable, Dict[Domain, float]]:
    #     """
    #     Calculate a fuzzy matrix (unordered default-dictionary).
    #     """
    #     result: Dict[FuzzyVariable, Dict[Domain, float]] = defaultdict(Dict[Domain, float])
    #     result = defaultdict(Dict[Domain, float])
    #     for variable in self.inp:
    #         result[variable] = defaultdict(float)
    #         for d in variable.domain:
    #             result[variable][d] = d.mf.get_value(inp[variable])
    #     return result

    def rule_evaluation(self, rule_map: Dict[FuzzyVariable, Dict[Domain, float]]) -> List[float]:
        """
        Evaluate fuzzy conditions based on assertions in conditions.
        """
        result: List[float] = []
        for rule in self.rules:
            # If there is only one assertion in condition, then we do not deal with fuzzy rule operations.
            if len(rule[0].conditions) == 1:
                variable_name = rule[0].conditions[0][0]
                domain_name = rule[0].conditions[0][1]
                # Find fuzzy variable that has the name as what rules specified.
                for fuzzy_val in self.inp:
                    if fuzzy_val.name == variable_name:
                        d = fuzzy_val.get_domain_by_name(domain_name)
                        temp_result = rule_map[fuzzy_val][d]
                        # Deal with the 'is not' situation.
                        result.append(1.0 - temp_result if rule[0].not_[0] else temp_result)
                        break
            # If there are more than one assertion in condition, then we deal with fuzzy rule operations.
            else:
                multi_temp_result = []
                for i in range(len(rule[0].conditions)):
                    variable_name = rule[0].conditions[i][0]
                    domain_name = rule[0].conditions[i][1]
                    for fuzzy_val in self.inp:
                        if fuzzy_val.name == variable_name:
                            d = fuzzy_val.get_domain_by_name(domain_name)
                            temp_result = rule_map[fuzzy_val][d]
                            # Deal with the 'is not' situation.
                            multi_temp_result.append(1.0 - temp_result if rule[0].not_[i] else temp_result)
                            break

                operator_sorted = []
                for k, v in rule[0].op.items():
                    operator_sorted.append([k[0], k[1], v])

                operator_sorted = sorted(operator_sorted, key=lambda x: x[1])
                max_layer_index = max(l[0] for l in operator_sorted)

                # Fuzzy rule operations.
                def fuzzy_rule_op(op: FuzzyRuleOperatorType, partial_arr: List):
                    if op == FuzzyRuleOperatorType.AND:
                        if self.and_method == AndMethod.MIN:
                            return np.min(np.array(partial_arr))
                        elif self.and_method == AndMethod.PROD:
                            pass
                    elif op == FuzzyRuleOperatorType.OR:
                        if self.or_method == OrMethod.MAX:
                            return np.max(np.array(partial_arr))
                        elif self.or_method == OrMethod.PROB:
                            pass

                temp = 0
                for i in range(max_layer_index, -1, -1):
                    for j in range(len(operator_sorted)):
                        if operator_sorted[j][0] == i:
                            temp = fuzzy_rule_op(operator_sorted[j][2], [multi_temp_result[j], multi_temp_result[j + 1]])
                            multi_temp_result[j] = temp
                            multi_temp_result[j + 1] = temp

                result.append(temp)

        return result

    def inference(self, x: List[float]) -> List:
        """
        Inference conclusion of each rule.
        """
        result: List[List] = []
        for rule in self.rules:
            # Get the fuzzy membership function of the conclusion
            variable_name = rule[1].conclusion[0]
            domain_name = rule[1].conclusion[1]
            d = self.out.get_domain_by_name(domain_name)

            # Make a list of input for defuzzification and plotting.
            x_list = np.linspace(self.out_min, self.out_max, self.resolution)
            y_list = list(map(d.mf.get_value, x_list))
            result.append(y_list)

        if self.inference_method == InferenceMethod.MIN:
            return [np.fmin(x[i], result[i]) for i in range(len(x))]
        else:
            pass

    def aggregation(self, conclusion: List[float]) -> List:
        if self.aggregation_method == AggregationMethod.MAX:
            return np.amax(np.array(conclusion), axis=0)
        else:
            pass
