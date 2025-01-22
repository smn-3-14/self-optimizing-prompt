import unittest

from unittest.mock import MagicMock

from main import split_data, data
import ga


class TestGa(unittest.TestCase):
    train_set, val_set, test_set = split_data(data.read_problems())

    def test_fitness(self):
        population = ["Ich heiße Marvin"]

        ga._ask_model = MagicMock(return_value="print('Hello World')")

        fitness = ga.fitness_function(population[0], self.val_set)
        print("Fitness:", fitness)

    def test_tournament(self):
        population = ["Ich heiße Marvin", "Test", "Foo", "Bar", "Baz"]
        fitness = [0.4, 0.1, 0.3, 0.267, 0.75]

        selection = ga.selection(population, fitness)
        print("Selection:", selection)


if __name__ == "__main__":
    unittest.main()
