# self-optimizing-prompt

This repository contains a Python implementation of **[Genetic Algorithm for Prompt Engineering with Novel Genetic Operators](https://ieeexplore.ieee.org/document/10488291)**. The algorithm optimizes text prompts for AI models by evolving and refining them over multiple generations.

## Features
- Implements a Genetic Algorithm (GA) to optimize text prompts
- Multi-threaded evaluation for efficiency
- Logging and visualization tools for monitoring progress
- Tests against **HumanEval** and **MMLU-Pro** benchmarks

## Installation

Install required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run the script with default settings:
```bash
python main.py
```

Additional command-line arguments:
- `--uni` : Use the Uni API
- `--code` : Run code benchmark
- `--model <model_name>` : Specify which model to use
- `--dir <directory>` : Set the output directory
- `--baseline` : Create a baseline

## Configuration
Before running the script, create a `.env` file using `.env.example` as a template. This file should contain necessary environment variables such as API keys and configurations.

## Output
Results are saved in the `output/` directory, including:
- Best-performing prompts
- Evolution history logs
- Performance metrics

## Contributing
Pull requests are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`feature-xyz`)
3. Commit your changes
4. Push to your branch and create a Pull Request

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Contact
For any questions, feel free to reach out via GitHub Issues.
