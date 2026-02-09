"""
UI Components
=============
Reusable custom widgets for the Apple Professional look.
"""

import tkinter as tk
from ui.styles import *

class HoverButton(tk.Button):
    """
    A minimalist button with Apple-style hover effects.
    """
    def __init__(self, master, **kwargs):
        self.default_bg = kwargs.get("bg", SIDEBAR_COLOR)
        self.default_fg = kwargs.get("fg", ACCENT_COLOR)
        self.hover_bg = kwargs.pop("hover_bg", "#2C2C2E") # Slightly lighter grey
        self.hover_fg = kwargs.pop("hover_fg", ACCENT_HOVER)
        
        # Set default styles if not provided
        if "bg" not in kwargs: kwargs["bg"] = self.default_bg
        if "fg" not in kwargs: kwargs["fg"] = self.default_fg
        if "font" not in kwargs: kwargs["font"] = FONT_BODY
        if "relief" not in kwargs: kwargs["relief"] = "flat"
        if "padx" not in kwargs: kwargs["padx"] = 16
        if "pady" not in kwargs: kwargs["pady"] = 8
        if "cursor" not in kwargs: kwargs["cursor"] = "hand2"
        if "bd" not in kwargs: kwargs["bd"] = 0
        
        super().__init__(master, **kwargs)
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self['background'] = self.hover_bg
        self['foreground'] = self.hover_fg
        
    def on_leave(self, e):
        self['background'] = self.default_bg
        self['foreground'] = self.default_fg

class CardFrame(tk.Frame):
    """
    A frame that looks like a card (dark bg, rounded-ish feel).
    """
    def __init__(self, master, **kwargs):
        if "bg" not in kwargs: kwargs["bg"] = CARD_BG
        if "bd" not in kwargs: kwargs["bd"] = 0
        if "relief" not in kwargs: kwargs["relief"] = "flat"
        
        super().__init__(master, **kwargs)
