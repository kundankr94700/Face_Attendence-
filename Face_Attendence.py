import pyttsx3 as t2s
import threading
from os import listdir,mkdir
from os.path import isfile, join
import cv2
import numpy as np
from csv import *
from tkinter import *
from tkinter.font import  Font
from time import *
from PIL import ImageTk, Image

face_cascade = cv2.CascadeClassifier('face_cas.xml')
def Recognise_Photo():
    eng2 = t2s.init()
    models=[]
    def training(data_path):
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        Training_Data, Labels = [], []
        for i, files in enumerate(onlyfiles):
            image_path = data_path + onlyfiles[i]
            images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(i)
        Labels = np.asarray(Labels, dtype=np.int32)
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(np.asarray(Training_Data), np.asarray(Labels))
        return model

    onlyfiles1 = [f for f in listdir('D:/Student Face Attendence/Student Records/')]

    for i in range((len(onlyfiles1) // 2) + 1):
        try:
            models.append(training('D:/Student Face Attendence/Student Records/Face_Data %d/' % (i + 1)))
        except:
            pass
    pre = 0
    model = models[0]
    csv_file = 'D:/Student Face Attendence/Student Records/Face_Record 1.csv'

    def text2speech_name(Id):
        try:
            eng2.setProperty('rate', 120)
            eng2.setProperty('volume', .9)
            eng2.say(Id)
            eng2.runAndWait()
        except:
            pass
    f12 = open('C:/Users/Kundan Kumar/OneDrive/Attendance  %s.csv' % asctime()[:11], 'a')
    f12.close()
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            j, k = model.predict(face)

            confidence = int(100 * (1 - (k) / 300))

            ref1, ref2 = j // 20, k // 10

            if (ref2 < 5):

                if confidence > 75:
                    Id = 'Searching'

                    f1 = open(csv_file, 'r')

                    r = reader(f1)
                    pos = []
                    for ii in r:

                        try:

                            if ref1 == int(ii[0]):
                                Id = ii[1]

                        except:
                            pass

                    f12 = open('C:/Users/Kundan Kumar/OneDrive/Attendance  %s.csv' % asctime()[:11], 'r')
                    r1 = reader(f12)
                    for iii in r1:
                        try:
                            pos.append(iii[0])
                        except:
                            pass
                    f12.close()

                    if Id not in pos:

                        cv2.putText(frame, 'Done ' + str(Id), (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 245, 0), 1)

                        try:
                            t = threading.Thread(name='child', target=text2speech_name, args=('Ok ' + Id,))
                            if not t.is_alive():
                                t.start()
                        except:
                            pass

                        f11 = open('C:/Users/Kundan Kumar/OneDrive/Attendance  %s.csv' % (asctime()[:11]),'a')
                        w = writer(f11)
                        if Id == 'Searching':
                            pass
                        else:
                            w.writerows([[Id, 'Present', asctime()]])
                        f11.close()
                    else:

                        cv2.putText(frame, 'Done ' + str(Id), (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 1,
                                    (0, 245, 0), 1)

                        try:
                            t = threading.Thread(name='child', target=text2speech_name,
                                                 args=(Id + 'is already present',))
                            if not t.is_alive():
                                t.start()
                        except:
                            pass
                else:
                    pass
            else:

                if pre < (len(onlyfiles1) // 2) - 1:
                    try:
                        model = models[pre + 1]
                        csv_file = 'D:/Student Face Attendence/Student Records/Face_Record %d.csv' % (pre + 2)

                    except:
                        pass
                    pre = pre + 1

                elif pre == (len(onlyfiles1) // 2) - 1:
                    try:
                        model = models[0]
                        csv_file = 'D:/Student Face Attendence/Student Records/Face_Record 1.csv'

                    except:
                        pass
                    pre = 0

                else:
                    pass
        cv2.putText(frame, 'Press Q to Exit', (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 125), 3)
        cv2.imshow('Face Recognition Software', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def add_Student():
    try:

        try:
            mkdir('D:/Student Face Attendence/Student Records/')
        except:
            pass

        def get_name():
            root_add.destroy()
            x112 = s11.get()

            def process_add(data_path, point):

                onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
                Labels = []
                for i, files in enumerate(onlyfiles):
                    Labels.append(i)
                Labels = np.asarray(Labels, dtype=np.int32)
                x2 = len(Labels)
                x1 = len(Labels) // 20
                name = x112

                f = open('D:/Student Face Attendence/Student Records/Face_Record %d.csv' % point, 'a')
                w = writer(f)

                w.writerows([[x1, name]])
                f.close()
                cap = cv2.VideoCapture(0)
                count = 0

                while True:
                    ret, frame = cap.read()

                    frame = cv2.flip(frame, 1)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    n = 0
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        n = faces.shape[0]
                        face_img = frame[y:y + h, x:x + w]

                    if n == 0:
                        cv2.putText(frame, 'No Face found', (10, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0),
                                    2)
                    else:

                        count += 1
                        face = cv2.resize(face_img, (200, 200))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                        cv2.imwrite('%s/user' % data_path + str(count + x2) + '.jpg', face)

                        cv2.putText(face, str(count), (5, 180), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face Id Capture', face)

                    cv2.imshow('Capture Face_Data', frame)
                    if cv2.waitKey(1) == ord('q') or count == 20:
                        break

                count = 0
                cap.release()
                cv2.destroyAllWindows()



            data_path = 'D:/Student Face Attendence/Student Records/'
            onlyfiles1 = [f for f in listdir(data_path)]
            xx = len(onlyfiles1) // 2 + 1
            try:
                mkdir('D:/Student Face Attendence/Student Records/Face_Data %d/' % (xx))
            except:
                pass

            process_add('D:/Student Face Attendence/Student Records/Face_Data %d/' %(xx), xx)

        root_add = Toplevel()
        root_add.geometry('430x450+700+100')
        img1 = ImageTk.PhotoImage(Image.open("123.png"))
        panel = Label(root_add, image=img1).place(x=60, y=10)
        l1 = Label(root_add, text='Face Recognition Software', font=f5, fg='Red').place(x=80, y=5)
        l1 = Label(root_add, text='Enter Name of the Student ', font=f5, fg='darkblue').place(x=80, y=330)
        s11 = StringVar()
        ent = Entry(root_add, textvariable=s11, font=f5).place(x=100, y=370)
        b = Button(root_add, text='Add Student Record', command=get_name, width=25, bg='Green', height=1, fg='white',
                   font=f3).place(x=70, y=410)
        c1 = Canvas(root_add, width=10, height=750, bg='Brown')
        c1.pack(side=RIGHT)
        c1 = Canvas(root_add, width=10, height=750, bg='Brown')
        c1.pack(side=LEFT)
        root_add.resizable('false', 'false')
        root_add.mainloop()
    except:
        pass

root_Face_Reognise = Tk()
root_Face_Reognise.geometry('750x460+600+100')
frame=Frame(root_Face_Reognise,height=600,width=235).pack(side=RIGHT)
img = ImageTk.PhotoImage(Image.open("face_rec2.jpg"))
panel = Label(frame, image = img)
panel.pack(side = "left", fill = "both", expand = "yes")
try:
    mkdir('D:/Student Face Attendence/')
except:
    pass
f3 = Font(family="Time New Roman", size=15, weight="bold",underline=1)
f5 = Font(family="Time New Roman", size=15, weight="bold")
f6 = Font(family="Time New Roman", size=12, weight="bold")
root_Face_Reognise.title('Face Attendence Software')
l3 = Label(root_Face_Reognise, text="Machine Learning ", fg='brown', font=f3).place(x=540, y=60)
l3 = Label(root_Face_Reognise, text="Training Project", fg='brown', font=f3).place(x=550, y=90)
l = Label(root_Face_Reognise, text='Face Attendence ', fg='darkblue', font=f3).place(x=550, y=150)
l = Label(root_Face_Reognise, text='Software', fg='darkblue', font=f3).place(x=590, y=180)
b1 = Button(root_Face_Reognise, text='Add Student Face', bg='skyBlue', fg='white', width=18, height=1, command=add_Student,font=f3).place(x=520, y=250)
b1 = Button(root_Face_Reognise, text='Make_Attendence', bg='white', width=18, height=1, command=Recognise_Photo, font=f3).place(x=520, y=310)
b1 = Button(root_Face_Reognise, text='Close', bg='green', width=18, height=1, command=root_Face_Reognise.destroy,font=f3).place(x=520, y=370)
Label(root_Face_Reognise, text="Copyright @ Kundan Kumar ", fg='darkblue',font=f6).place(x=10, y=420)
root_Face_Reognise.resizable(False, False)
root_Face_Reognise.mainloop()
