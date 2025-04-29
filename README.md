
# PINNs Model

This project sets up a Python development environment in Visual Studio Code (VSCode) for running a Physics-Informed Neural Network (PINNs) on our magnet problem.

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/rahavdayan/pinns-model.git
cd pinns-model
```

### 2. Open in VSCode

- Open the cloned project folder (`pinns-model`) in [Visual Studio Code](https://code.visualstudio.com/).

---

## ⚙️ Setting up the Python Development Environment

Follow the official [VSCode Python Tutorial](https://code.visualstudio.com/docs/python/python-tutorial) to install Python support.

You can use either **standard Python** or **Anaconda**.

### 3. Create and Activate a Virtual Environment

If you don't already have a `.venv`, create one:

```bash
python -m venv .venv
```

Activate it:

- **Windows**:
  ```bash
  .\.venv\Scriptsctivate
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

---

### 4. Install Dependencies

Once the virtual environment is activated, install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧩 VSCode Extensions

Install the following extensions from the VSCode Marketplace:

- **Python** (by Microsoft)
- **Jupyter** (by Microsoft)

> Tip: Open the Extensions tab (`Ctrl+Shift+X`), search for the extension names, and install them.

---

## 📒 Running Jupyter Notebooks

To properly run `.ipynb` files:

1. Open a Jupyter Notebook file in VSCode.
2. At the top right, click **"Select Kernel"**.
3. Click **"Select Another Kernel"**.
4. Choose the environment located inside your `.venv` folder.
5. Click **"Run All"** at the top to execute all cells.
6. If prompted, click **"Install"** to set up any required Jupyter tools.

> **Note**: Jupyter Notebook files are currently listed in `.gitignore` to prevent version conflicts between collaborators.

---

## 📌 Notes

- Always make sure your virtual environment is **activated** while working on the project.
- If you encounter missing dependencies during development, install them individually using:

```bash
pip install <package-name>
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/rahavdayan/pinns-model.git
cd pinns-model
python -m venv .venv
# Activate your environment
# Install dependencies
pip install -r requirements.txt
# Open in VSCode and set kernel for Jupyter notebooks
```

---

## 🛠️ How the Code Works

### File Structure

```
└── src (source folder for project) 
    ├── cooling (example PINNs from Medium article) 
    ├── ferrofluid
    │   ├── dimensional (our final dimensionalized approach)  
    │   ├── droplet_data (data to train the model on for various droplet sizes) 
    │   │   ├── experimental_data
    │   │   └── paper_data
    │   ├── nondimensional (our old nondimensionalized approach) 
    └── toy_problem (a PINNs we ran on a toy problem)
```

The folder you should focus on is `dimensional`. All the other folders may be non-functional. Inside the `dimensional` folder, there are three main files:

- **`diff_equation.py`**: Defines the differential equation and associated parameters.
- **`network.py`**: Defines the PINNs network architecture.
- **`ferrofluid.ipynb`**: Jupyter notebook that runs the network.

### Key Parameters in Code

#### `diff_equation.py`
Defines key parameters for the physics-based model:

- `dom_ext`: Percentage by which the domain of evaluation of the physics loss is extended from the domain of the training data.
- `num_eval_points`: Number of evaluation points used when determining the physics loss.

#### `ferrofluid.ipynb`
Defines key parameters for the training process:

- `num_epochs`: Number of epochs for training.
- `data_loss_weight`: Weight of the data loss in the loss function.
- `physics_loss_weight`: Weight of the physics loss in the loss function.
- `lr`: Learning rate (recommended: `batch_size * 1e-4`).
- `batch_size`: Size of each training batch.

### Modifying Training Data

If you need to change which droplet size files you use, modify the `grab_training_data()` function in `diff_eq.py`. You will need to change the `r` and `V` variables accordingly.

### Running the Code

Once you have configured everything as needed (parameters, data, etc.), simply run the Jupyter notebook `ferrofluid.ipynb` to see your results!

---

## 📚 Credit

This approach is borrowed from the article [Physics-Informed Neural Networks: A Simple Tutorial with PyTorch](https://medium.com/@theo.wolf/physics-informed-neural-networks-a-simple-tutorial-with-pytorch-f28a890b874a) by Theo Wolf on Medium.
