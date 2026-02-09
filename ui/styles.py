"""
UI Styles and Constants
=======================
Defines the color palette, fonts, and layout constants for the Apple Professional UI.
"""

# Sea Blue Vibrant Palette
BG_COLOR = "#0A1929"        # Deep Sea Blue
SIDEBAR_COLOR = "#0D2847"   # Ocean Blue
CARD_BG = "#132F4C"         # Lighter Sea Blue
HEADER_COLOR = "#0A1929"    # Deep Sea Blue    

# Apple System Colors
APPLE_BLUE = "#0A84FF"      # System Blue
APPLE_GREEN = "#30D158"     # System Green
APPLE_INDIGO = "#5E5CE6"    # System Indigo
APPLE_ORANGE = "#FF9F0A"    # System Orange
APPLE_PINK = "#FF375F"      # System Pink
APPLE_PURPLE = "#BF5AF2"    # System Purple
APPLE_RED = "#FF453A"       # System Red
APPLE_TEAL = "#64D2FF"      # System Teal
APPLE_YELLOW = "#FFD60A"    # System Yellow

# Semantic Colors
ACCENT_COLOR = APPLE_BLUE
ACCENT_HOVER = "#409CFF"    # Lighter Blue
TEXT_COLOR = "#F5F5F7"      # Apple White
TEXT_DIM = "#86868B"        # Apple Grey (Subtext)
SUCCESS_COLOR = APPLE_GREEN
ERROR_COLOR = APPLE_RED
WARNING_COLOR = APPLE_ORANGE

# Fonts (Apple Style)
# Windows doesn't have SF Pro by default, use Segoe UI or Helvetica Neue
FONT_FAMILY = "Segoe UI" 
FONT_HEADER = (FONT_FAMILY, 14, "bold")
FONT_TITLE = (FONT_FAMILY, 32, "bold") # Large, Hero text
FONT_BODY = (FONT_FAMILY, 11)
FONT_SMALL = (FONT_FAMILY, 9)
FONT_CLUE = (FONT_FAMILY, 18, "bold")

# Dimensions
SIDEBAR_WIDTH = 240
HEADER_HEIGHT = 60
CELL_SIZE = 60
DOT_RADIUS = 3
