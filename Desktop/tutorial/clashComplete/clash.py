from pyautogui import *
import pyautogui
import time
import keyboard
import random
import random
import win32api, win32con
def click(x,y):
    win32api.SetCursorPos((x+random.randint(0, 5) ,y+random.randint(0, 5)))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.1) #This pauses the script for 0.1 seconds
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

def drag(x1,y1,x2,y2):
    win32api.SetCursorPos((x1,y1))
    time.sleep(0.03)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.07)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

    time.sleep(0.07)
    win32api.SetCursorPos((x2,y2))
    time.sleep(0.03)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.06)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

def randboardx():
    return random.randint(600, 1040)



def randboardy():
    return random.randint(100, 600)

def checkgame():
    try:
        location = pyautogui.locateOnScreen('ingame.png',region = (540,40,570,400), grayscale = True, confidence=0.4)
        
        if location is not None:    
             return False
        else:
            print("game not detected")
            return True

    except pyautogui.ImageNotFoundException:
         
         print("game not detected")
         return True
     
    except Exception as e:
        print(f"Unexpected error: {e}")
        time.sleep(0.5)
        return False

def checkok():
    try:
        location = pyautogui.locateOnScreen('okbut.png',region = (750,830,160,100), grayscale = True, confidence=0.7)
        
        if location is not None:    
             click(location[0]+5,location[1]+5)
             print("found ok button")
             time.sleep(3)
        else:
            print("----")
            return True

    except pyautogui.ImageNotFoundException:
         
         print("----")
         return True
     
    except Exception as e:
        print(f"Unexpected error: {e}")
        time.sleep(0.5)


def battle(x,y,width,height):
    targ = 'target.png'
    print("setup")
    time.sleep(13)
    print("starting")
 #ok button   region = (750,830,160,100)
    while checkok():
        if checkgame():
            print("breaking out")
            time.sleep(10)
            break
        
        if keyboard.is_pressed('q'):
            break
        try:
             tlocation = pyautogui.locateOnScreen(targ,region = (660,830,450,120), grayscale = True, confidence=0.6)
             
             
             
             if tlocation is not None:
                 print("found him")
                 drag(tlocation[0]+random.randint(5, 15),tlocation[1]+random.randint(5, 15),randboardx(),randboardy())
                 
                 time.sleep(4.2)
                 
             else:
                 print("placing extra")
            
                 rando = random.randint(1, 4)
                 if rando == 2 or rando == 1:
                    drag(random.randint(740, 760),random.randint(875, 895),randboardx(),randboardy())
                 if rando == 3 :
                    drag(random.randint(925, 935),random.randint(875, 895),randboardx(),randboardy())
                 if rando == 4 :
                    drag(random.randint(995, 1007),random.randint(875, 895),randboardx(),randboardy())
                    
                 time.sleep(5)
                 
         
        except pyautogui.ImageNotFoundException:
             
            print("placing extra")
            
            rando = random.randint(1, 4)
            if rando == 2 or rando == 1:
                drag(random.randint(740, 760),random.randint(875, 895),randboardx(),randboardy())
            if rando == 3 :
                drag(random.randint(925, 935),random.randint(875, 895),randboardx(),randboardy())
            if rando == 4 :
                drag(random.randint(995, 1007),random.randint(875, 895),randboardx(),randboardy())
                
            time.sleep(5)
            
         
        except Exception as e:
            print(f"Unexpected error: {e}")
            time.sleep(0.5)


    
count = 0
while keyboard.is_pressed('q') == False:
    x,y = 540,40
    width, height = 570,900
    pic = pyautogui.screenshot(region = (x,y,width,height))
    try:
         location = pyautogui.locateOnScreen('battle.png',region = (x,y,width,height), grayscale = True, confidence=0.7)
         if location is not None:
             click(location[0],location[1])
             count = 0
             print("starting")
             battle(x,y,width,height)
         else:
             time.sleep(2)
             print("not starting")
     
    except pyautogui.ImageNotFoundException:
         
        time.sleep(2)
        print("not starting")
        count += 1
        if count>=10 :
            checkok()
            count =0
     
    except Exception as e:
        print(f"Unexpected error: {e}")
        time.sleep(2)
    
