# README Instructions

## Usage Instructions

### 1. Running the Main Script

```bash
./script.sh <PLACE> <METHOD> <MODEL>
```

**Example:**

```bash
./script.sh "Los Angeles, California, USA" "b" "O"
```

---

### 2. Method Options

| Code | Description                  |
|------|------------------------------|
| `b`  | Betweenness Centrality       |
| `c`  | Closeness Centrality         |
| `s`  | Straightness Centrality      |
| `i`  | Information Centrality       |

---

### 3. Model Options

| Code | Description                  |
|------|------------------------------|
| `O`   | Organised City               |
| `SO`  | Self-Organised City          |

---

### 4. Running `random_generated_graphs.py`

This script is **not executable**, so run it with Python directly:

```bash
python random_generated_graphs.py
```

---

### Notes

- Make sure you have activated your virtual environment if needed.  
- Ensure all dependencies (`networkx`, `osmnx`, etc.) are installed.

