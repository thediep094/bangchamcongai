# Face Detection and Face Recognition in Personal's Check-in and Check-out

In this first version, I use MTCNN (lightweight) and Facenet512 to solve the task. If possible, I will develop more pair in the next version (RetinaFace + ArcFace, etc). 

Note: the performance when use only CPU shows that MTCNN lightweight has the highest fps (much better than MTCNN, RetinaFace).


## 1. Run requirement.txt to install necessary libraries.
Run in terminal: 

`pip install -r requirements.txt`

## 2. Automatically capture new faces

### 2.1. By hand
This step is optional. You can create and put your photos in a folder with the name is your name in `./MTCNN_Facenet512/data/test_images`.

![faces](readme_images/Screenshot%20from%202023-04-25%2010-30-10.png)

For example, I have folder `An` which is my name and it contains my pictures.

### 2.2. Automatically 
If don't want to do it by hand, just run `./core_AI/get_face.py`

In terminal, type your name. A pop-up will show up and automatically captures, detects and crops 10 pictures of your face. You can change the number of pictures in variable `count` in line 11.

![name](readme_images/Screenshot%20from%202023-04-25%2010-39-33.png)

## 3. Update facelist

This step will zip all the folder contains name and pictures in step 2. Just run `./core_AI/update_facelist.py`. 

## 4. Detection

All set! Run `./core_AI/face_recognition.py`. Depend on your hardware, it will detect real face in real-time with high or low fps.

## 5. Development

I make and create a database to save people information in and out time on MongoDB. This is not my major, so contact me so we could colab together about it.

### 5.1. Start-up backend

Run `/home/minelove/Face Detection/data_server/app.py`

You will be able to access your localhost. Click `Add User` to add new users.

After that, click on `Update`. It will automatically add the face and the ID to our database and server.

### 5.2. Start-up Face Recognition

Similar as step 4, you will run `data_server/face_recognition.py` to auto detection the users that you added them in 5.1.

### 5.3. Finished

Run `processing_data/process_data.py` to save the check-in and check-out data of users to your database.

In this case, I use:

- Database: MongoDB Atlas
- Backend: Python
- Frontend: HTML.


If you can't run, feel free to contact me!

### Hope my small project could help you!

<sub>*from minelove with love*<sub>

