# Loopy - Slitherlink Puzzle Game
## Project Overview
Loopy is a logic puzzle game played on a rectangular lattice of dots. The objective is to connect adjacent dots with horizontal and vertical lines to form a single continuous loop, without any crossings or branches. The numbers in the squares indicate how many of the four surrounding sides must be part of the loop.

This project is submitted as part of the Design and Analysis of Algorithms (DAA) coursework to demonstrate the practical application of fundamental algorithms in game development.

---

## Phase 1: Current Implementation
This phase focuses on the core game mechanics and the implementation of Greedy Algorithms and Sorting techniques to manage game states, generate puzzles, provide hints, and save progress.

### Project Structure
- `main.py`: The entry point of the application.
- `ui/`: Contains all Tkinter-based user interface code.
  - `main_window.py`: Manages the main application window and navigation.
  - `pages.py`: Handles individual screens like Home, Game, and Statistics.
  - `styles.py`: Defines the visual theme (colors, fonts).
- `logic/`: Contains the core game logic.
  - `game_state.py`: Manages the state of the board, turns, undo/redo stack, and interaction with algorithms.
  - `graph.py`: Represents the grid as a graph data structure.
  - `validators.py`: Checks for valid moves and winning conditions.
  - `statistics.py`: Tracks wins and losses.
- `daa/`: Contains the specific algorithm implementations required for the course.
  - `sorting.py`: Implementation of sorting algorithms.
  - `greedy_algos.py`: Implementation of Standard Greedy Algorithms.
  - `graph_algos.py`: Standard graph traversals (BFS, DFS) for validation.

### Technical Specifications
- **Language**: Python 3.x
- **GUI Framework**: Tkinter
- **Data Structures**: Graphs (Adjacency Lists), Priority Queues (Heap), Disjoint Sets (Union-Find), Trees.

### Algorithm Implementation Details

#### 1. Sorting Algorithms
We implemented five standard sorting algorithms to order candidate moves for the AI and to support Greedy strategies.
- **Bubble Sort, Insertion Sort, Selection Sort**: These O(N^2) algorithms are implemented to sort small datasets, such as the list of edges connected to a single node when determining the next move.
- **Merge Sort, Quick Sort**: These O(N log N) algorithms are used when sorting larger lists, such as the entire set of board edges based on their weights or heuristic scores. In the Greedy CPU, moves are sorted by their "value" (heuristic score) to determine the best immediate step.

#### 2. Greedy Strategy
The core of the project's complexity lies in the application of Greedy Algorithms. Below is the usage of each:

- **Fractional Knapsack (Energy Management)**:
  In "Expert Mode," the player and CPU have a limited "Energy" resource. Each edge on the board is assigned a weight (cost). The goal is to maximize the number of correct moves (value) without exceeding the energy limit (capacity). The Fractional Knapsack approach is used conceptually to evaluate the "value-to-weight" ratio of edges, prioritizing low-cost edges that have a high probability of being correct.

- **Prim's Algorithm (Puzzle Generation)**:
  We use Prim's algorithm to generate the puzzle board. The algorithm starts from a random cell and grows a Minimum Spanning Tree (MST) on the grid of cells. The edges of this MST effectively define a connected region. The boundary of this region becomes the "solution loop" for the puzzle. This ensures that the generated loop is always continuous and valid.

- **Kruskal's Algorithm (Validation)**:
  Kruskal's algorithm, using a Disjoint Set (Union-Find) data structure, is used to prevent cycles in the "Greedy CPU" logic when it is trying to connect loose ends without closing the loop prematurely. It helps in maintaining the forest of line segments before they form the final single loop.

- **Dijkstra's Algorithm (Hint System)**:
  The "Smart Hint" feature uses Dijkstra's algorithm. When a player asks for a hint, the system often identifies two "loose ends" (vertices with degree 1) on the board. Dijkstra's is then run to find the shortest path between these two ends through the graph of valid empty edges. This provides the most efficient path to connect disconnected segments.

- **Huffman Coding (Save/Load Game)**:
  To demonstrate data compression, we use Huffman Coding for the Save/Load functionality. The game's Undo Stack (a history of all moves) is converted into a string. We calculate the frequency of characters in this string, build a Huffman Tree, and generate optimal prefix codes. The save file stores the compressed binary data, significantly reducing file size compared to raw text.

---

## Phase 2: Future Enhancements
The current implementation relies heavily on Greedy approaches, which make locally optimal choices. Phase 2 will introduce more advanced paradigms to solve complex board states and optimize performance.

### 1. Divide and Conquer
The Greedy approach for solving or generating puzzles can be slow on very large grids.
- **Parallel Solvers**: We can use Divide and Conquer to split a large board into smaller non-overlapping regions (sub-grids), solve them independently, and then merge the solutions. This would significantly speed up the valid move checking and puzzle generation for large (e.g., 20x20) boards.
- **Optimized Sorting**: While we currently have Merge/Quick sort, we can integrate them more deeply into the game state management, for example, to maintain a sorted structure of "most constrained cells" (cells with 0 or 3 remaining lines) to instantly identify forced moves.

### 2. Dynamic Programming (DP)
Greedy algorithms do not always guarantee a global optimum or a solution in constraint-heavy scenarios.
- **Solving Constraint Conflicts**: The Greedy CPU sometimes gets stuck in local optima where no valid move exists but the game isn't lost. A Dynamic Programming approach (e.g., typically used in "broken profile" DP for grid connectivity problems) could be implemented to look ahead. We can define a state as the configuration of the "boundary" line passing through the grid rows. By storing the results of subproblems (valid partial loops for the first k rows), we can determine if a valid solution exists without backtracking blindly.
- **Hint System Enhancement**: Currently, Dijkstra finds the shortest path of *valid* moves, but it doesn't know if that path will eventually lead to a deadlock. DP can be used to compute the "Longest Path" or specific connectivity patterns that ensure the loop remains closeable.
- **New Mode: "Optimization Challenge"**: A new game mode where the player must complete the loop with a specific total weight (exact subset sum). This is a variation of the Knapsack problem that requires DP (0/1 Knapsack) rather than the Greedy (Fractional) approach, as we cannot take "half an edge."

### Conclusion of Phase 1
The current application successfully demonstrates the utility of Greedy algorithms in a practical, interactive environment. The game is playable, the AI provides a reasonable challenge, and the underlying systems for saving and generation are robustly backed by standard algorithms.
