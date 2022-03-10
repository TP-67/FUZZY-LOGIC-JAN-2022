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

# Define membership function
# Wind speed
w1 = Domain('slow', ZMF(5.0, 25.0))
w2 = Domain('fast', SMF(10.0, 36.0))

# Temperature
t1 = Domain('low', GaussianMF(3.0, 4.0))
t2 = Domain('medium', GaussianMF(20.0, 4.0))
t3 = Domain('high', GaussianMF(30.0, 4.0))

# Humidity
h1 = Domain('dry', AlterLinearMF(15.0, 1.0, 30.0, 0))
h2 = Domain('fair', TriangularMF(15.0, 30.0, 45.0))
h3 = Domain('wet', AlterLinearMF(30.0, 0, 45.0, 1.0))

# Mountain range
m1 = Domain('huge', SigmoidMF(50, -0.2))
m2 = Domain('hill', SigmoidMF(45, 0.7))

# Raining
r1 = Domain('hardly', SigmoidMF(25, -0.7))
r2 = Domain('certain', SigmoidMF(25, 0.3))

# Convert to fuzzy variables
wind: FuzzyVariable = FuzzyVariable('wind', 0, 1, w1, w2)
temperature: FuzzyVariable = FuzzyVariable('temperature', 0, 1, t1, t2, t3)
humidity: FuzzyVariable = FuzzyVariable('humidity', 0, 1, h1, h2, h3)
mountain: FuzzyVariable = FuzzyVariable('mountain', 0, 1, m1, m2)
raining: FuzzyVariable = FuzzyVariable('raining', 0, 1, r1, r2)

# Define constant value
wind_x_min = 0
wind_x_max = 50
wind_x_resolution = (wind_x_max - wind_x_min) * 5
temperature_x_min = 0
temperature_x_max = 50
temperature_x_resolution = (temperature_x_max - temperature_x_min) * 5
humidity_x_min = 0
humidity_x_max = 50
humidity_x_resolution = (humidity_x_max - humidity_x_min) * 5
mountain_x_min = 0
mountain_x_max = 100
mountain_x_resolution = (mountain_x_max - mountain_x_min) * 5
raining_x_min = 0
raining_x_max = 50
raining_x_resolution = (raining_x_max - raining_x_min) * 5

if not check_valid([wind_x_min, wind_x_max],
                   [temperature_x_min, temperature_x_max],
                   [humidity_x_min, humidity_x_max],
                   [mountain_x_min, mountain_x_max],
                   [raining_x_min, raining_x_max]):
    raise Exception('Invalid Input Values')

# Define rules
rule = Rule()
conditions1, conclusion1 = rule.parse_rule('if (wind is fast) and (temperature is low) then (raining is hardly)')
conditions2, conclusion2 = rule.parse_rule('if (wind is slow) and (humidity is wet) then (raining is certain)')
conditions3, conclusion3 = rule.parse_rule('if (temperature is medium) and (humidity is wet) then (raining is certain)')
conditions4, conclusion4 = rule.parse_rule('if (temperature is high) and (humidity is dry) then (raining is hardly)')
conditions5, conclusion5 = rule.parse_rule('if ((wind is slow) and (mountain is huge)) or ((humidity is dry) and (mountain is hill)) then (raining is hardly)')
conditions6, conclusion6 = rule.parse_rule('if (temperature is not low) and (humidity is not dry) then (raining is certain)')

# Initial fuzzy engine
fuzzy_system = FuzzyEngine([wind, temperature, humidity, mountain], raining, FuzzyRuleOperatorType.AND, raining_x_min, raining_x_max)

# Add fuzzy rules
fuzzy_system.rules.append([conditions1, conclusion1])
fuzzy_system.rules.append([conditions2, conclusion2])
fuzzy_system.rules.append([conditions3, conclusion3])
fuzzy_system.rules.append([conditions4, conclusion4])
fuzzy_system.rules.append([conditions5, conclusion5])
fuzzy_system.rules.append([conditions6, conclusion6])

# Fuzzy inference
fuzzy_matrix, conditions, conclusions, fuzzy_result = fuzzy_system.calculate(OrderedDict({wind: 25.0, temperature: 28.0, humidity: 33.0, mountain: 32.0}))

# Defuzzification
defuzzifier_m = Mamdani(raining_x_min, raining_x_max)
d_m = defuzzifier_m.get_value(fuzzy_result)
s_m = "Anticipated raining index (Mamdani): " + str(d_m)
print(s_m)

defuzzifier_t = Tsukamoto([InverseSigmoidMF(25, -0.7),
                           InverseSigmoidMF(25, 0.3),
                           InverseSigmoidMF(25, 0.3),
                           InverseSigmoidMF(25, -0.7),
                           InverseSigmoidMF(25, -0.7),
                           InverseSigmoidMF(25, 0.3)])
d_t = defuzzifier_t.get_value(conclusions, raining_x_min, raining_x_max)
s_t = 'Anticipated raining index (Tsukamoto):' + str(d_t)
print(s_t)

# Print results to txt file
write_text(config, s_m, s_t)

# Plot membership function with input values
plot_mf(wind, 25.0, wind_x_min, wind_x_max, wind_x_resolution, config)
plot_mf(temperature, 28.0, temperature_x_min, temperature_x_max, temperature_x_resolution, config)
plot_mf(humidity, 33.0, humidity_x_min, humidity_x_max, humidity_x_resolution, config)
plot_mf(mountain, 32.0, mountain_x_min, mountain_x_max, mountain_x_resolution, config)

# Plot individual fuzzy rules
plot_conclusions(raining, raining_x_min, raining_x_max, raining_x_resolution, conclusions, config)

# Plot the fuzzy result rule
plot_fuzzy_result(raining_x_min, raining_x_max, raining_x_resolution, fuzzy_result, config)

# Plot the fuzzy result rule with defuzzified centroid
plot_fuzzy_result_with_center(raining_x_min, raining_x_max, raining_x_resolution, fuzzy_result, d_m, config)


if __name__ == '__main__':
    pass
