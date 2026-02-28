"""
Board Canvas
============
Handles the drawing of the game grid, edges, and interaction.
Clean, minimalist Apple style.
"""

import tkinter as tk
from ui.styles import *

class BoardCanvas(tk.Canvas):
    def __init__(self, master, game_state, on_move_callback):
        try:
            with open(r"c:\Users\LAKSHMI NARAYANA\Desktop\Loopy_SimonTatham_Games\debug_full.log", "a") as f:
                f.write("BoardCanvas init called\n")
        except:
            pass
        super().__init__(master, bg=BG_COLOR, highlightthickness=0)
        self.game_state = game_state
        self.on_move_callback = on_move_callback
        
        self.cell_size = CELL_SIZE
        self.margin = 40
        
        self.bind("<Button-1>", self.on_click)
        self.bind("<Motion>", self.on_hover)
        self.bind("<Configure>", self.on_resize)
        
        self.hovered_edge = None
        self.hint_edge = None
        
        # Initialize coordinates to avoid AttributeError before first draw
        self.start_x = 0
        self.start_y = 0
        
    def on_resize(self, event):
        self.draw()

    def draw(self):
        self.delete("all")
        
        rows = self.game_state.rows
        cols = self.game_state.cols
        
        # Center the grid
        width = self.winfo_width()
        height = self.winfo_height()
        
        if width < 10: width = 600
        if height < 10: height = 600
            
        grid_w = cols * self.cell_size
        grid_h = rows * self.cell_size
        
        self.start_x = (width - grid_w) // 2
        self.start_y = (height - grid_h) // 2
        
        # Draw Background Grid (Very subtle dots)
        for r in range(rows + 1):
            for c in range(cols + 1):
                x = self.start_x + c * self.cell_size
                y = self.start_y + r * self.cell_size
                # Tiny guide dots
                self.create_oval(x-1, y-1, x+1, y+1, fill="#3A3A3C", outline="")

        # Draw Clues
        for r in range(rows):
            for c in range(cols):
                if (r, c) in self.game_state.clues:
                    val = self.game_state.clues[(r, c)]
                    x = self.start_x + c * self.cell_size + self.cell_size // 2
                    y = self.start_y + r * self.cell_size + self.cell_size // 2
                    
                    color = TEXT_DIM # Default grey
                    if self._is_clue_satisfied((r, c), val):
                        color = APPLE_BLUE # Blue when done (Apple style)
                    elif self._is_clue_violated((r, c), val):
                        color = APPLE_RED
                        
                    self.create_text(x, y, text=str(val), font=FONT_CLUE, fill=color)
        
        # Draw Active Edges
        for edge in self.game_state.graph.edges:
            color = TEXT_COLOR
            width = 3
            # Highlight the last CPU move in yellow
            if self.game_state.last_cpu_move_info and edge == self.game_state.last_cpu_move_info.get("move"):
                color = APPLE_YELLOW
                width = 4
                
            self._draw_edge(edge, color, width=width)
            
        # Draw Hovered Edge
        if self.hovered_edge:
            edge = tuple(sorted(self.hovered_edge))
            if edge not in self.game_state.graph.edges:
                self._draw_edge(self.hovered_edge, "#3A3A3C", width=3) # Dark grey hover

        # Draw Hint Edge
        if self.hint_edge:
            # Use stored hint color if available, else default to yellow (though show_hint defaults it too)
            color = getattr(self, 'hint_color', APPLE_YELLOW) 
            self._draw_edge(self.hint_edge, color, width=3)

        # Draw Dots (Vertices)
        for r in range(rows + 1):
            for c in range(cols + 1):
                x = self.start_x + c * self.cell_size
                y = self.start_y + r * self.cell_size
                self.create_oval(x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS, fill=TEXT_COLOR, outline="")

        # Draw Weights (All Modes)
        if self.game_state.edge_weights:
            for edge, weight in self.game_state.edge_weights.items():
                u, v = edge
                r1, c1 = u
                r2, c2 = v
                x1 = self.start_x + c1 * self.cell_size
                y1 = self.start_y + r1 * self.cell_size
                x2 = self.start_x + c2 * self.cell_size
                y2 = self.start_y + r2 * self.cell_size
                
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                
                # Offset slightly
                if r1 == r2: # Horizontal
                    my -= 10
                else: # Vertical
                    mx -= 10
                    
                self.create_text(mx, my, text=str(weight), font=FONT_SMALL, fill=TEXT_DIM)

        # Draw Cognitive Overlay (Reasoning Layer)
        self._draw_overlay_elements()

    def _draw_edge(self, edge, color, width):
        (r1, c1), (r2, c2) = edge
        x1 = self.start_x + c1 * self.cell_size
        y1 = self.start_y + r1 * self.cell_size
        x2 = self.start_x + c2 * self.cell_size
        y2 = self.start_y + r2 * self.cell_size
        
        self.create_line(x1, y1, x2, y2, width=width, fill=color, capstyle=tk.ROUND)

    def _is_clue_satisfied(self, cell, val):
        from logic.validators import count_edges_around_cell
        return count_edges_around_cell(self.game_state.graph, cell) == val

    def _is_clue_violated(self, cell, val):
        from logic.validators import count_edges_around_cell
        return count_edges_around_cell(self.game_state.graph, cell) > val

    def on_hover(self, event):
        edge = self._get_closest_edge(event.x, event.y)
        if edge != self.hovered_edge:
            self.hovered_edge = edge
            # self.draw() # Optimization: Only draw if needed? 
            # Actually, draw() is fast enough usually.
            
            # TEACHER MODE
            if self.game_state.teacher_mode and self.hovered_edge:
                self.check_teacher_mode(self.hovered_edge)
            else:
                self.draw()

    def check_teacher_mode(self, edge):
        """
        Highlight edge RED if it's a bad move.
        """
        # Simple heuristic check for "Bad Move" explanation
        u, v = edge
        if (tuple(sorted(edge)) in self.game_state.graph.edges):
            self.draw() # Removing edge - usually fine
            return
            
        # Check against basic rules
        from logic.validators import count_edges_around_cell
        is_bad = False
        reason = ""
        
        # 1. 0-clue rule
        # ...reuse logic? simplified here for UI speed
        adj_cells = self._get_adj_cells(u, v)
        for cell in adj_cells:
            if cell in self.game_state.clues:
                if self.game_state.clues[cell] == 0:
                    is_bad = True
                    reason = "Bad Move: Violates '0' clue!"
                    
        self.draw()
        if is_bad:
            self._draw_edge(edge, APPLE_RED, width=3)
            # Show tooltip text at mouse pos?
            # For simplicity, we create text on canvas near edge
            mx, my = self._get_midpoint(u, v)
            self.create_text(mx, my - 15, text=reason, fill=APPLE_RED, font=FONT_SMALL)

    def _get_adj_cells(self, u, v):
        r1, c1 = u
        r2, c2 = v
        graph = self.game_state.graph
        adj_cells = []
        if r1 == r2: # Horizontal
            c_min = min(c1, c2)
            if r1 > 0: adj_cells.append((r1-1, c_min))
            if r1 < graph.rows: adj_cells.append((r1, c_min))
        else: # Vertical
            r_min = min(r1, r2)
            if c1 > 0: adj_cells.append((r_min, c1-1))
            if c1 < graph.cols: adj_cells.append((r_min, c1))
        return adj_cells
        
    def _get_midpoint(self, u, v):
        r1, c1 = u
        r2, c2 = v
        x1 = self.start_x + c1 * self.cell_size
        y1 = self.start_y + r1 * self.cell_size
        x2 = self.start_x + c2 * self.cell_size
        y2 = self.start_y + r2 * self.cell_size
        return (x1+x2)/2, (y1+y2)/2

    def on_click(self, event):
        if self.game_state.game_over: return
        
        # If CPU turn, ignore clicks?
        if "CPU" in self.game_state.turn: return
        
        edge = self._get_closest_edge(event.x, event.y)
        if edge:
            u, v = edge
            self.on_move_callback(u, v)
            self.hint_edge = None
            self.draw()

    def _get_closest_edge(self, x, y):
        threshold = 15
        rows = self.game_state.rows
        cols = self.game_state.cols
        
        best_edge = None
        min_dist = float('inf')
        
        # Horizontal
        for r in range(rows + 1):
            for c in range(cols):
                x1 = self.start_x + c * self.cell_size
                y1 = self.start_y + r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1
                
                if abs(y - y1) < threshold and x1 <= x <= x2:
                    dist = abs(y - y1)
                    if dist < min_dist:
                        min_dist = dist
                        best_edge = ((r, c), (r, c+1))
                        
        # Vertical
        for r in range(rows):
            for c in range(cols + 1):
                x1 = self.start_x + c * self.cell_size
                y1 = self.start_y + r * self.cell_size
                x2 = x1
                y2 = y1 + self.cell_size
                
                if abs(x - x1) < threshold and y1 <= y <= y2:
                    dist = abs(x - x1)
                    if dist < min_dist:
                        min_dist = dist
                        best_edge = ((r, c), (r+1, c))
                        
        return best_edge

    def show_hint(self, edge, color=APPLE_YELLOW):
        self.hint_edge = edge
        self.hint_color = color
        self.draw()

    # ------------------------------------------------------------------
    # COGNITIVE VISUALIZATION LAYER
    # ------------------------------------------------------------------
    def show_reasoning_overlay(self, explanation):
        """
        Display the cognitive overlay:
        - Scope highlight (Region/Global/Local)
        - Decided edge highlight
        """
        self.current_overlay = explanation
        self.draw()

    def clear_overlay(self):
        self.current_overlay = None
        self.draw()

    def _draw_overlay_elements(self):
        if not hasattr(self, "current_overlay") or not self.current_overlay:
            return

        expl = self.current_overlay
        
        # 1. Region Highlight
        if expl.scope == "Global":
            # Faint purple over whole board
            self._draw_region_rect((0, 0, self.game_state.rows-1, self.game_state.cols-1), "#6f42c1", alpha=0.1)
        
        elif expl.scope == "Regional" and expl.highlight_region:
            # Faint blue over specific region
            self._draw_region_rect(expl.highlight_region, "#17a2b8", alpha=0.15)

        elif expl.scope == "Local" and expl.highlight_cells:
            # Highlight specific cells involved
            for r, c in expl.highlight_cells:
                self._draw_cell_highlight(r, c, "#28a745", alpha=0.2)

        # 2. Edge Highlight (Strong)
        if expl.highlight_edges:
            for edge in expl.highlight_edges:
                self._draw_edge(edge, APPLE_YELLOW, width=5)

    def _draw_region_rect(self, region, color, alpha):
        # Tkinter doesn't support alpha directly on canvas shapes easily without PIL or stippling.
        # We'll use a stipple pattern to simulate transparency or just a colored outline + thin crossover.
        # For a "Cognitive" feel, let's use a nice outline and a very light stipple if possible,
        # or just a distinct outline convention.
        
        # Simulating "Faint" fill with stipple 'gray25' or similar if available, 
        # but cross-platform stipples are flaky. 
        # Let's use a hollow rectangle with thick borders for now, or lines.
        
        r_min, c_min, r_max, c_max = region
        
        x1 = self.start_x + c_min * self.cell_size
        y1 = self.start_y + r_min * self.cell_size
        x2 = self.start_x + (c_max + 1) * self.cell_size
        y2 = self.start_y + (r_max + 1) * self.cell_size
        
        # Draw soft background (stippled)
        self.create_rectangle(x1, y1, x2, y2, fill=color, outline="", stipple="gray12", tags="overlay")
        # Draw strong border
        self.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="overlay")

    def _draw_cell_highlight(self, r, c, color, alpha):
        x1 = self.start_x + c * self.cell_size
        y1 = self.start_y + r * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        self.create_rectangle(x1, y1, x2, y2, fill=color, outline="", stipple="gray25", tags="overlay")

    # ------------------------------------------------------------------
    # AI VISUALIZATION FEATURES
    # ------------------------------------------------------------------
    
    def visualize_cpu_thinking(self, candidates, best_move, final_callback):
        """
        Show top candidates in Yellow, then pick Green.
        """
        # 1. Show Top 3 Candidates
        self.draw() # Clear previous
        
        # Sort candidates to look meaningful (best first) or random?
        # candidates is list of (move, score)
        top_candidates = candidates[:3]
        
        for move, score in top_candidates:
            self._draw_edge(move, APPLE_YELLOW, width=4)
            
        self.update() # Force redraw
        
        # 2. Wait 500ms
        self.after(500, lambda: self._finalize_cpu_move(best_move, final_callback))
        
    def _finalize_cpu_move(self, best_move, callback):
        # 3. Highlight Chosen Move in Green
        self.draw()
        self._draw_edge(best_move, APPLE_GREEN, width=4)
        self.update()
        
        # 4. Wait 200ms then execute
        self.after(200, lambda: callback())

    def play_victory_animation(self):
        """
        Confetti effect!
        """
        colors = [APPLE_RED, APPLE_GREEN, APPLE_BLUE, APPLE_ORANGE, APPLE_PURPLE, APPLE_TEAL, APPLE_YELLOW]
        width = self.winfo_width()
        height = self.winfo_height()
        
        for _ in range(50):
            x = __import__("random").randint(0, width)
            y = __import__("random").randint(0, height // 2) # Start from top half
            color = __import__("random").choice(colors)
            size = __import__("random").randint(5, 10)
            
            self.create_oval(x, y, x+size, y+size, fill=color, outline="")
            
        # Schedule another burst? Or just one
        # Let's do a simple loop for 1 second
        # Using simple method for now.

