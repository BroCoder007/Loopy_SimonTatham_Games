"""
Loopy DAA Project - Phase 1
===========================
Entry point for the application.
"""

import ctypes
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass # Not on Windows or already set

from ui.main_window import MainWindow

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
