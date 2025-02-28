""" Code for final evaluation and graphs """

import json
from pathlib import Path

import matplotlib.pyplot as plt

from shared import get_max, get_avg, get_min


base_path = Path("/home/simon/Documents/Uni/seminar_prompt_engineering/src/results/1738575778_hf_qa_C0gavFmI")
generations = 9

def load_population(file: Path):
    with file.open() as f:
        return json.load(f)

eval = []
max_test = []
min_test = []
avg_test = []
x = []

for generation in range(generations):
    x.append(generation)
    file = base_path / f"population_{generation}.json"
    population = load_population(file)
    max_test.append(get_max(population)["fitness"])
    min_test.append(get_min(population)["fitness"])
    avg_test.append(get_avg(population))

    file = base_path / f"eval_{generation}.json"
    population = load_population(file)
    eval.append(population["fitness"])

print("Max:", max_test[-1])
print("Min:", min_test[-1])
print("Average:", avg_test[-1])
# print("Eval:", eval)

baseline_path = base_path / "base_line.json"
baseline = load_population(baseline_path)

print("Baseline:", baseline["fitness"])

plt.figure(figsize=(8, 5))
plt.plot(x, [baseline["fitness"]] * generations , linestyle='-', label='Baseline')
plt.plot(x, max_test, marker='o', linestyle='-', label='Best')
plt.plot(x, min_test, marker='s', linestyle='--', label='Worst')
plt.plot(x, avg_test, marker='^', linestyle='-.', label='Median')

plt.xlabel('Generation')
plt.ylabel('Pass@1')
plt.legend()
plt.grid(True)

# Show the plot
# plt.show()

