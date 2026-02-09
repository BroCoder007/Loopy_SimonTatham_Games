"""
Audio Module
============
Simple sound effects helper using winsound.
Gracefully fails on non-Windows systems.
"""

import sys

def play_sound(event):
    """
    Play a sound based on the event type.
    Events: 'move', 'error', 'win', 'click'
    """
    if sys.platform != "win32":
        return

    try:
        import winsound
        
        if event == 'move':
            # Low freq beep
            winsound.Beep(400, 100)
        elif event == 'error':
            # Low long beep
            winsound.Beep(150, 300)
        elif event == 'win':
            # Victory fanfare (simple sequence)
            winsound.Beep(523, 150) # C5
            winsound.Beep(659, 150) # E5
            winsound.Beep(784, 300) # G5
        elif event == 'click':
            winsound.Beep(800, 50)
            
    except Exception:
        pass
