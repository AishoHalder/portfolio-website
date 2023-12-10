import cv2
import numpy as np
from PIL import Image
import os

class imp:
    def generate_dataset ():
                id_P = "Aisha"
                
                #=============Load predifined data=================
                face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                
                def Face_cropped(img):
                    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    face = face_classifier.detectMultiScale(gray,1.3,5) #scaling factor = 1.3 ; Minimum neighbour = 5;
                    
                    for (x,y,w,h) in face:
                        face_cropped = img[y:y+h,x:x+w]
                        return face_cropped
                # (0) = own camera (url) for other .
                cap = cv2.VideoCapture(0)
                img_id = 0
                while True:
                    ret,My_frame = cap.read()
                    if Face_cropped(My_frame) is not None :
                        img_id+=1
                        face = cv2.resize(Face_cropped(My_frame),(450,450))
                        face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
                        file_name_path = "data/user."+str(id_P)+"."+str(img_id)+".jpg"
                        cv2.imwrite(file_name_path,face)
                        cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
                        cv2.imshow("croped face",face)
                    
                    if cv2.waitKey(1)==13 or int(img_id)==1000:
                        break
                cap.release()
                cv2.destroyAllWindows()
    def train_classifier():
        data_dir = ("data")
        path = [os.path.join(data_dir,file) for file in os.listdir(data_dir)]
        
        faces = []
        ids = []
        
        for image in path:
            img = Image.open(image).convert('L')
            imageNp = np.array(img,'uint8')
            id = str(os.path.split(image)[1].split('.')[1])
            id = int(id.replace("Aisha",""))
            faces.append(imageNp)
            ids.append(id)
            cv2.imshow("Training",imageNp)
            cv2.waitKey(1)==13
        ids = np.array(ids)
        
        #==================Train the classifier and save=================
        clf = cv2.face.LBPHFaceRecognizer.create()
        clf.train(faces,ids)
        clf.write("classifier.xml")
        cv2.destroyAllWindows()

    def face_recog():
        xyz = 0
        def draw_boundary(xyz,img,classifier,scaleFactor,minNeighbors,color,text,clf):
            gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
            
            coord = []
            for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                id,predict = clf.predict(gray_image[y:y+h,x:x+w])
                confidence = int((100*(1-predict/300)))
                id = str(id)
                nid = "Aisha"
                if confidence>77:
                    cv2.putText(img,f'Name : {nid}',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    xyz+=1
                else:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(img,f'Unknown',(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                
                coord = [x,y,w,h]
            return coord,xyz
        def recognize(xyz,img,clf,faceCascade):
            coord,xyz = draw_boundary(xyz,img,faceCascade,1.1,10,(255,25,255),"Face",clf)
            return img,xyz
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer.create()
        clf.read("classifier.xml")
        # (0) for defaiult (1) for connected camera (url) for cctv camera
        video_cap = cv2.VideoCapture(0)
        
        while True :
            ret,img = video_cap.read()
            img,xyz = recognize(xyz,img,clf,faceCascade)
            cv2.imshow("Face Recognition",img)
            
            if cv2.waitKey(1)==13 or xyz == 13 :
                break
        video_cap.release()
        cv2.destroyAllWindows()
        return xyz


        