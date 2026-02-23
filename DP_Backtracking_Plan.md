# Loopy Project Upgrade: DP & Backtracking Execution Plan

This document outlines the final project upgrade for the Loopy game, ensuring all 3 teammates get hands-on experience with Dynamic Programming (DP) and Backtracking. The plan is designed to **guarantee NO merge conflicts** for Git beginners and ensures everyone easily gets **9+ commits**.

---

## 🏗️ 1. Work Distribution & Feature Plan

To avoid merge conflicts, the golden rule is: **Everyone works in their own entirely separate files.**

### 🧑‍💻 Person 1: The Team Leader (You) 
**Focus:** The Core Brain & Main UI Integration (Most Important & Scoring, Hardest Task)
**Concept:** **Memoized Backtracking Solver & Solver UI** 
You will build the primary AI that solves the puzzle, combining Backtracking with **Dynamic Programming (State Memoization)** to exponentially reduce time complexity. 
You will also build the UI panel that controls the solver (e.g., "Solve Step-by-Step", "Solve Instantly", and speed sliders).
* **Workspace (Logic):** Create a new file `logic/solvers/dp_backtracking_solver.py`
* **Workspace (UI):** Create a new file `ui/solver_control_panel.py`
* **Why it scores high:** Combining DP as a cache within recursive Backtracking is complex, and integrating it with real-time UI updates shows full-stack capability.

### 🧑‍🤝‍🧑 Person 2: Teammate A
**Focus:** Intelligent Hint Engine, Validator & Hint UI
**Concept:** **DP Shortest Path, Validation Backtracking & Visual Feedback**
Use a Dynamic Programming algorithm to suggest the next move, and a Backtracking function to validate if the current state can lead to a win.
You will also build the visual UI components that display these hints (e.g., glowing edges, error highlights when a move is invalid).
* **Workspace (Logic):** Create a new file `logic/dp_hints_engine.py`
* **Workspace (UI):** Create a new file `ui/hint_visualizer.py`
* **Why it matters:** Extremely visible to the end-user. Tying a complex backend algorithm directly to a visual UI component is highly rewarding.

### 🧑‍🤝‍🧑 Person 3: Teammate B
**Focus:** Puzzle Generation, DP Analytics & Level Select UI
**Concept:** **Generative Backtracking, DP Path Counting & Menu Systems**
Use Backtracking to *generate* a valid puzzle grid, then apply a DP counting algorithm to calculate its "Difficulty Score".
You will also build the "New Game / Level Select" UI screen that lets the user choose the difficulty, which hooks into your generator.
* **Workspace (Logic):** Create a new file `logic/generators/dp_generator.py`
* **Workspace (UI):** Create a new file `ui/level_selection_menu.py`
* **Why it matters:** Proves understanding of DP for combinatorics and quantitative analysis, plus builds out the core user onboarding flow.

---

## 🌳 2. Git Workflow (How to Avoid ALL Merge Conflicts)

Because you are all beginners to GitHub, merge conflicts happen when two people edit the **same lines of the same file**. Our strategy makes this **impossible**.

### 1. The Core Rule: Sequential Commits
To absolutely guarantee there are no merge conflicts, the team will **never work on the code at the exact same time**. Instead, the team will work in phases, committing straight to the `main` branch one person after another.

### 2. The Rotation Order
1.  **Phase 1: Person 3 (Teammate B)** builds the Puzzle Generator & Level Select Menu. They push their 9+ commits to `main`.
2.  **Phase 2: Person 2 (Teammate A)** pulls the updated `main`, builds the Hint Engine & Visualizer, and pushes their 9+ commits to `main`.
3.  **Phase 3: Person 1 (Team Leader)** pulls the updated `main`, builds the Core Solver & Control Panel, and pushes their commits.
4.  **Phase 4: Person 1 (Team Leader)** wires all the completed UI pieces together into the main game container and makes the final commit.

### 3. Write Code in YOUR Files ONLY
Each person only creates and edits their assigned new file mentioned above. **Do not touch `main.py` or existing UI files during this phase.**

### 4. Getting 9+ Commits Easily
Make a "commit" every time you add a small piece of logic. Do not wait until the end to commit! 
* *Commit 1:* `Added base class and function signatures`
* *Commit 2:* `Implemented the backtracking base case`
* *Commit 3:* `Added DP cache dictionary`
* *Commit 4:* `Added state serialization string method`
* *Commit 5:* `Integrated loop logic`
* *Commit 6:* `Added helper functions for neighbor detection`
* *Commit 7:* `Wrote simple test print statements`
* *Commit 8:* `Added algorithm time complexity comments`
* *Commit 9:* `Final bug fixes for the module`

### 5. Step-by-Step Commit Guide (For Your Turn)
When it is your phase to code, follow these exact steps:
1.  Open your terminal and make sure you are on main: `git checkout main`
2.  Get the latest code from the person who went before you: `git pull origin main`
3.  **Create your assigned files** (e.g., `dp_hints_engine.py`). **Do not touch existing shared files like `main.py`.**
4.  Write a small piece of logic.
5.  Commit right away: `git add .` then `git commit -m "Added basic logic structure"`
6.  Repeat steps 4 and 5 until your feature is fully built and you have your 9+ commits.
7.  Push all your commits to GitHub: `git push origin main`
8.  Tell the next person it is their turn to `git pull`!

---

**Summary for your Professor/Scorer:**
* All 3 teammates utilized Backtracking.
* All 3 teammates utilized Dynamic Programming.
* The workloads are perfectly distributed, mathematically avoiding Git overlaps.
* The Leader handles the hardest algorithmic challenge (recursive state memoization) and leads the final integration.
