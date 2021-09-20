from flask import Flask, render_template, request, redirect, url_for, send_file
import os

#we are importing the function that makes predictions.
from model import predict_dt

import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True


#we are setting the name of our html file that we added in the templates folder
@app.route('/')
def index():
    return render_template('index.html')

#this is how we are getting the file that the user uploads. 
#then we are setting the path that we want to save it so we can use it later for predictions
@app.route("/", methods=['POST'])
def uploadFiles():
    uploaded_file = request.files['file']
    
    if uploaded_file.filename != '':
        file_path = ( "/Users/jaadeoye/Desktop/PythonDT_SMOTE/static/files/file.csv")
        uploaded_file.save(file_path)
    return redirect(url_for('downloadFile'))

#now we are reading the file, make predictions with our model and save the predictions.
#then we are sending the CSV with the predictions to the user as attachement 
@app.route('/download')
def downloadFile ():
    path = "/Users/jaadeoye/Desktop/PythonDT_SMOTE/static/files/file.csv"
    predictions=predict_dt(pd.read_csv(path))
    predictions.to_csv('/Users/jaadeoye/Desktop/PythonDT_SMOTE/static/files/predictions.csv',index=False)
    return send_file("/Users/jaadeoye/Desktop/PythonDT_SMOTE/static/files/predictions.csv", as_attachment=True)

#here we are setting the port. 
if (__name__ == "__main__"):
     app.run(debug=True,host='0.0.0.0', port=9080)
