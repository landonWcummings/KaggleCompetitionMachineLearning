from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con


while keyboard.is_pressed('q') == False:
    pos = pyautogui.position()
    print(pos[0])
    print(pos[1])
    color = pyautogui.pixel(pos[0], pos[1])
    print(color[0])
    print(color[1])
    print(color[2])
    print("-----")
    time.sleep(1)
    
    
