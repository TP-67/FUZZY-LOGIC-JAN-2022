import os
import sys
sys.path.append(os.path.abspath('../src'))

import yaml
import numpy as np
from collections import OrderedDict

from domain import *
from variable import *
from mf import *
from rule import *
from fuzzy_engine import *
from defuzzifier import *
from inv_mf import *
from utils import *


# Load configuration
with open('../config.yml', 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Create output folder
abs_dir = os.path.abspath('../output')
file_name = os.path.splitext(__file__)[0]
output_dir = os.path.join(abs_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
config['output_path'] = output_dir

# Define membership functions for both input and output fuzzy variables.
# If we use Sugeno method, we need to define membership functions separately.
a1: Domain = Domain('less', AlterLinearMF(30, 1.0, 70, 0))
a2: Domain = Domain('more', AlterLinearMF(40, 0, 80, 1.0))

h1: Domain = Domain('good', SigmoidMF(25, 0.3))
h2: Domain = Domain('poor', SigmoidMF(25, -0.7))

# Define fuzzy variables for both input and output fuzzy variables.
# If we use Sugeno method, we need to define output fuzzy variable separately.
alc: FuzzyVariable = FuzzyVariable('alc', 0, 1, a1, a2)
health: FuzzyVariable = FuzzyVariable('health', 0, 1, h1, h2)

# Define membership functions for Sugeno method.
r1: SugenoFuzzyFunction = SugenoFuzzyFunction('high', {alc: 0.1, health: 0.4}, 0.5)
r2: SugenoFuzzyFunction = SugenoFuzzyFunction('low', {alc: 0.4, health: 0.2}, 0.7)

# Define a output fuzzy variable for Sugeno method.
rate: SugenoFuzzyVariable = SugenoFuzzyVariable('rate', r1, r2)

# Define boundary values
alc_x_min = 0
alc_x_max = 100
alc_x_resolution = (alc_x_max - alc_x_min) * 5
health_x_min = 0
health_x_max = 50
health_x_resolution = (health_x_max - health_x_min) * 5
rate_x_min = 0
rate_x_max = 100

# Check for validation
if not check_valid([alc_x_min, alc_x_max], [health_x_min, health_x_max], [rate_x_min, rate_x_max]):
    raise Exception('Invalid Input Values')

# Define rules
# Note: if we use Sugeno defuzzification method, there can not be more than two assertions in conditions.
rule = Rule()
conditions1, conclusion1 = rule.parse_rule('if (alc is less) and (health is good) then (rate is high)')
conditions2, conclusion2 = rule.parse_rule('if (alc is more) and (health is poor) then (rate is low)')

# Initial fuzzy engine
fuzzy_system = FuzzyEngine([alc, health], rate, FuzzyRuleOperatorType.AND, rate_x_min, rate_x_max)

# Add fuzzy rules
fuzzy_system.rules.append([conditions1, conclusion1])
fuzzy_system.rules.append([conditions2, conclusion2])

# Fuzzy inference
# For Mamdani and Tsukamoto methods, we call 'calculate' function.
# For Sugeno method, we call 'calculate_sugeno' function.
fuzzy_matrix, conditions, fuzzy_result = fuzzy_system.calculate_sugeno(OrderedDict({alc: 65.0, health: 34.0}))

# Defuzzification
# Note:
#   (1) If we set a SugenoFuzzyVariable as our output variable, then we can only defuzzify with Sugeno defuzzification.
#   (2) If we set a FuzzyVariable as our output variable, then we can always use Mamdani defuzzification.
#   (3) If we set a FuzzyVariable as our output variable, and the membership functions of the output variable is strictly increasing or decreasing, then we can use both Mamdani and Tsukamoto defuzzification.
defuzzifier_m = Sugeno()
d_s = defuzzifier_m.get_value(conditions, fuzzy_result)
s_s = "Anticipated health index (Sugeno): " + str(d_s)
print(s_s)

# Print results to txt file
write_text(config, s_s)

# Plot membership function with input values
plot_mf(alc, 65.0, alc_x_min, alc_x_max, alc_x_resolution, config)
plot_mf(alc, 34.0, health_x_min, health_x_max, health_x_resolution, config)

# Plot individual fuzzy rules
# We only call this function for Mamdani and Tsukamoto defuzzification.
# Pass

# Plot the aggregated fuzzy result rule
# We only call this function for Mamdani and Tsukamoto defuzzification.
# Pass

# Plot the fuzzy result rule with the defuzzified centroid
# We only call this function for Mamdani and Tsukamoto defuzzification.
# Pass


if __name__ == '__main__':
    pass
