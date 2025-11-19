import pyautogui
import time

def capture_ui():
    time.sleep(2)
    pyautogui.screenshot('Z:/corpus_platform/logs/ui_snapshot.png')
    return True
