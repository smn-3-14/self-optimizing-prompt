import json
from pathlib import Path

from shared import get_max, get_avg, get_min


base_path = Path("/home/simon/Documents/Uni/seminar_prompt_engineering/src/results/1738578476_uni_code_7Umhs6AX")

def load_population(file: Path):
    with file.open() as f:
        return json.load(f)

eval = []
max_test = []
min_test = []
avg_test = []

for i in range(9):
    file = base_path / f"population_{i}.json"
    population = load_population(file)
    max_test.append(get_max(population)["fitness"])
    min_test.append(get_min(population)["fitness"])
    avg_test.append(get_avg(population))

    file = base_path / f"eval_{i}.json"
    population = load_population(file)
    eval.append(population["fitness"])

print("Max:", max_test)
print("Min:", min_test)
print("Average:", avg_test)
print("Eval:", eval)

baseline_path = base_path / "base_line.json"
baseline = load_population(baseline_path)

print("Baseline", baseline["fitness"])

