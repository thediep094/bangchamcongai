import mtcnn
import cv2

def get_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = mtcnn.MTCNN()
    
    faces = detector.detect_faces(img)
    print('hello')
    for face in faces:
        x, y, w, h = face['box']
        #crop
        cropped = img[y:y+h, x:x+w]
        return cropped

if __name__ == '__main__':
    img = cv2.imread('/home/minelove/Face Detection/1_frame2.jpg')
    img = get_face(img)
    #show img
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()