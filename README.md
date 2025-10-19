# approximate-nearest-neighbor-graphs

## Overview

This repository contains implementations and benchmarks for my Thesis on **approximate nearest neighbor (ANN) graph algorithms**. ANN graphs are used to efficiently find approximate nearest neighbors in high-dimensional spaces, making them useful for applications like recommendation systems, image retrieval, and clustering.

## Features

- Efficient construction of ANN graphs.
- Support for various distance metrics (e.g., Euclidean, cosine similarity).
- Benchmarking tools to compare performance across datasets.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required libraries listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/approximate-nearest-neighbor-graphs.git
    cd approximate-nearest-neighbor-graphs
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the main script to construct and query an ANN graph:
```bash
python main.py --dataset path/to/dataset --metric euclidean
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
