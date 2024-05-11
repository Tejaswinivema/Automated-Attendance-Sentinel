import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import pyttsx3
import tkinter as tk

# Initialize Flask app
app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Initializing VideoCapture object to access Webcam
face_detector = cv2.CascadeClassifier('C:\\Users\\nikhi\\OneDrive\\Desktop\\teju\\Major Project\Automated-Attendance-Sentinel\haarcascade_frontalface_default.xml')

# If these directories don't exist, create them
directories = ['Attendance', 'static', 'static/faces']
for directory in directories:
    if not os.path.isdir(directory):
        os.makedirs(directory)

# Create CSV file for today's attendance if it doesn't exist
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Roll-No,Name,Branch,Time')

# Initialize a dictionary to store user data including points or scores
users = {}

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in extract_faces: {e}")
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    try:
        return model.predict(facearray)[0]
    except IndexError:
        return "User not recognized"

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    rolls = df['Roll-No']
    names = df['Name']
    branches = df['Branch']
    times = df['Time']
    l = len(df)
    return rolls, names, branches, times, l

def add_attendance(name, branch):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    # Check if the user attended on time (you can define your criteria here)
    # For example, let's assume attendance is considered on time if it's before 9:30 AM
    on_time = datetime.now().time() < datetime.strptime("09:30:00", "%H:%M:%S").time()

    # Update the user's points or scores based on attendance timeliness
    if on_time:       
        # Voice-based notification for reward
        engine.say(f'Attendance captured for roll number {userid}')
        engine.runAndWait()

    # Update attendance record
    df = pd.read_csv(attendance_file)
    if str(userid) not in df['Roll-No'].values:
        with open(attendance_file, 'a') as f:
            f.write(f'\n{userid},{username},{branch},{current_time}')
        
        # Voice-based notification
        engine.say(f'User registered bearing roll number {userid}')
        engine.runAndWait()

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    branches = []
    l = len(userlist)

    for i in userlist:
        name, roll, branch = i.split('_')
        names.append(name)
        rolls.append(roll)
        branches.append(branch)

    return userlist, rolls, names, branches, l

def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(os.path.join(duser, i))
    os.rmdir(duser)

# Tkinter authentication window
def authenticate():
    root = tk.Tk()
    root.title("Authentication")
    
    # Set the size of the window
    root.geometry("700x350")

    # Set background color
    root.config(bg="#f0f0f0")

    # Add padding
    root.padding = 20

    # Set font style
    font_style = ("Helvetica", 12)

    def login():
        correct_username = "admin"
        correct_password = "password"

        username = entry_username.get()
        password = entry_password.get()

        # Check if the entered username and password match predefined credentials
        if username == correct_username and password == correct_password:
            root.destroy()  # Close the Tkinter window if authenticated
            app.run(debug=True)  # Start the Flask app
        else:
            # Show an error message if authentication fails
            label_error.config(text="Invalid username or password")
    
    def on_closing():
        root.destroy()  # Close the Tkinter window

    label_username = tk.Label(root, text="Username:", bg="#f0f0f0", font=font_style)
    label_username.pack(pady=(10, 0))
    entry_username = tk.Entry(root, font=font_style)
    entry_username.pack()

    label_password = tk.Label(root, text="Password:", bg="#f0f0f0", font=font_style)
    label_password.pack(pady=(10, 0))
    entry_password = tk.Entry(root, show="*", font=font_style)
    entry_password.pack()

    button_login = tk.Button(root, text="Login", command=login, bg="#4CAF50", fg="white", font=font_style)
    button_login.pack(pady=(20, 0))

    label_error = tk.Label(root, text="", fg="red", bg="#f0f0f0", font=font_style)
    label_error.pack(pady=(10, 0))

    # Bind the closing event to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.lift()
    root.mainloop()

# Call the authentication window before running the Flask app
#authenticate()

# Our main page
@app.route('/')
def home():
    rolls, names, branches, times, l = extract_attendance()
    return render_template('home.html', rolls=rolls, names=names, branches=branches, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## List users page
@app.route('/listusers')
def listusers():
    userlist, rolls, names, branches, l = getallusers()
    return render_template('listusers.html', userlist=userlist, rolls=rolls, names=names, branches=branches, l=l, totalreg=totalreg(), datetoday2=datetoday2)

## Delete functionality
@app.route('/deleteuser', methods=['GET'])
def deleteuser():
    duser = request.args.get('user')
    deletefolder(os.path.join('static/faces', duser))

    ## if all the face are deleted, delete the trained file...
    if not os.listdir('static/faces/'):
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except Exception as e:
        print(f"Error in train_model: {e}")

    userlist, rolls, names, branches, l = getallusers()
    return render_template('listusers.html', userlist=userlist, rolls=rolls, names=names, branches=branches, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    rolls, names, branches, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', rolls=rolls, names=names, branches=branches, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            branch = "CSE-DS"  # Replace this with the actual branch value

            if identified_person == "User not recognized":
                # Handle unrecognized user
                cv2.putText(frame, "User not recognized", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Add attendance for recognized user
                add_attendance(identified_person, branch)  # Pass the branch value to add_attendance
                cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    rolls, names, branches, times, l = extract_attendance()
    return render_template('home.html', rolls=rolls, names=names, branches=branches, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['POST'])
def add():
    newuserid = request.form['newuserid']
    newusername = request.form['newusername']
    newbranch = request.form['newbranch']
    userimagefolder = os.path.join('static/faces', f'{newusername}_{newuserid}')
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i = 0
    cap = cv2.VideoCapture(0)
    while i < 7:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{7}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            name = f'{newusername}_{i}.jpg'
            cv2.imwrite(os.path.join(userimagefolder, name), frame[y:y+h, x:x+w])
            i += 1
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    rolls, names, branches, times, l = extract_attendance()
    add_attendance(f'{newusername}_{newuserid}', newbranch)  # Add attendance with correct branch value

    return render_template('home.html', rolls=rolls, names=names, branches=branches, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, newbranch=newbranch)

# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
