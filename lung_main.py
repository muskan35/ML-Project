import tkinter as tk
from tkinter import *
from PIL import Image, ImageOps, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import time
import imutils

window = tk.Tk()
window.title("Lung Detection")

window.geometry('1100x650')
window.configure(background='gainsboro')



message = tk.Label(window, text="Lung Cancer Detection",bg= 'dark cyan', fg="white", width=48,
                   height=2, font=('times', 30, 'bold '))
message.place(x=5, y=10)

def exit():
        window.destroy()

def on_closing():
    from tkinter import messagebox
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        window.destroy()
window.protocol("WM_DELETE_WINDOW", on_closing)

#Disable scientific notation for clarity
np.set_printoptions(suppress=True)


# Loading image

net = cv2.dnn.readNet("data/obj.weights","data/obj.cfg")
classes = []
with open("data/obj.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def model_predict():
    frame = cv2.imread(path)
    #loading image
    font = cv2.FONT_HERSHEY_PLAIN
    height,width,channels = frame.shape

    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #reduce 416 to 320    
    net.setInput(blob)
    outs = net.forward(outputlayers)

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    TrackedIDs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)                
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            print(label)
            if(label == "Tumor"):
                color = (255,0,0)
            else:
                color = (0,204,0)
            confidence= confidences[i]         
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            
            cv2.putText(frame,label +str(round(confidence,2)),(x,y),font,0.8,(0,0,255),1)
    frame = imutils.resize(frame,width = 400)
    cv2.imshow("Image",frame)
    key = cv2.waitKey(0) #wait 1ms the loop will start again and we will process the next frame    
    cv2.destroyAllWindows()


def openphoto():
    global path
    path=askopenfilename(filetypes=[("Image File",'')])
    im = cv2.imread(path)
    cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    cv2image = imutils.resize(cv2image, width=350)
    img = Image.fromarray(cv2image)
    tkimage = ImageTk.PhotoImage(img)
    myvar=tk.Label(window,image = tkimage, height="350", width="350")
    myvar.image = tkimage
    myvar.place(x=390,y=150)
    preImg = tk.Button(window, text="Predict",fg="white",command=model_predict ,bg="dark cyan"  ,width=20  ,height=3, activebackground = "medium turquoise" ,font=('times', 15, ' bold '))
    preImg.place(x=90, y=270)

takeImg = tk.Button(window, text="Select Image",command=openphoto,fg="white"  ,bg="dark cyan"  ,width=20  ,height=3, activebackground = "medium turquoise" ,font=('times', 15, ' bold '))
takeImg.place(x=90, y=150)

quitWindow = tk.Button(window, text="Quit", command=on_closing  ,fg="white"  ,bg="dark cyan"  ,width=20  ,height=3, activebackground = "medium turquoise" ,font=('times', 15, ' bold '))
quitWindow.place(x=800, y=450)

window.mainloop()
