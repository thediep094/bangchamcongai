from flask import Flask, request, render_template, redirect, url_for, session
from pymongo import MongoClient
import uuid
from werkzeug.utils import secure_filename
import os
# from func.get_face import get_face_aws, check_exist
# from func.update_facelist import update_facelist, update_face_vector
import shutil
import boto3
import numpy as np
from functools import wraps
from convert_vector import convert_vector




app = Flask(__name__)
app.secret_key = 'notanymore'

# mongodb
url = "mongodb+srv://vudinhtruongan:demo@test.6o5s3eu.mongodb.net/test"
client = MongoClient(url)
db = client['user_management1']
collection = db['users']

# aws s3
aws_access_key_id = 'AKIAUJUMLJ3YNYNZ5KU6'
aws_secret_access_key = '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'
s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
bucket_name = 'truongan912'


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Main page
@app.route('/')
@login_required
def index():
    users = list(collection.find())
    return render_template('index.html', users=users)

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Perform authentication check, e.g., against database records
        if username == 'admin' and password == 'password':
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    else:
        return render_template('login.html')

# Logout page
@app.route('/logout')
@login_required
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))



@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    random_id = uuid.uuid4()
    if request.method == 'POST':
        id = request.form['id']
        name = request.form['name']
        birthdate = request.form['birthdate']
        sex = request.form['sex']
        address = request.form['address']
        email = request.form['email']
        file = request.files['photo']
        s3_filename = f"{id}.jpg"
        s3.upload_fileobj(file, bucket_name, s3_filename)
        embeds, id_ = convert_vector()
        id_ = np.char.rstrip(id_, '.jpg')
        index = np.where(id_ == id)[0]
        if len(index) > 0:
            target_value = embeds[index[0]].tolist()
        else:
            print("Không tìm thấy id trong mảng 2.")
        data = {
            'id': id,
            'name': name,
            'sex': sex,
            'birthdate': birthdate,
            'address': address,
            'email': email,
            'face': target_value
        }
        collection.insert_one(data)
        return redirect(url_for('index'))
    else:
        return render_template('add_user.html', random_id=random_id)


# edit user
@app.route('/edit_user/<string:id>', methods=['GET', 'POST'])
def edit_user(id):
    user = collection.find_one({'id': id})
    if request.method == 'POST':
        id = request.form['id']
        name = request.form['name']
        birthdate = request.form['birthdate']
        sex = request.form['sex']
        address = request.form['address']
        email = request.form['email']
        file = request.files['photo']
        s3_filename = f"{id}.jpg"
        s3.upload_fileobj(file, bucket_name, s3_filename)
        embeds, id_ = convert_vector()
        id_ = np.char.rstrip(id_, '.jpg')
        index = np.where(id_ == id)[0]
        if len(index) > 0:
            target_value = embeds[index[0]].tolist()
        else:
            print("Không tìm thấy id trong mảng 2.")
        collection.update_one({'id':id}, {'$set': {'name': name, 'sex': sex, 'birthdate': birthdate, 'address': address, 'email': email, 'face': target_value}})
        return redirect(url_for('index'))
    else:
        return render_template('edit_user.html', user=user)

# delete user
@app.route('/delete_user/<string:id>', methods=['GET', 'POST'])
def delete_user(id):
    folder = collection.find_one({'id': id})
    if folder:
        s3_filename = f"{id}.jpg"
        collection.delete_one({'id': id})
        s3.delete_object(Bucket=bucket_name, Key=s3_filename)
        return redirect(url_for('index'))
    else:
        return "Folder not found", 404



if __name__ == '__main__':
    app.run(debug=True)
