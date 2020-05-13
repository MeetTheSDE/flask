from flask import Flask ,render_template ,request
import numpy as np
from sklearn.externals import joblib
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predictcall",methods=['POST'])
def predictcall():
    if request.method == 'POST':
        x=request.form['message']
        sam=[x]
        vector = joblib.load('vectorizer.sav')
        ridge = joblib.load('model_ridge_class.sav')
        text = vector.transform(sam)
        result = np.round(ridge.predict(text))
        result=str(result).strip('[.]')
        return render_template('index.html',result=result)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)



'''from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import pickle
import sklearn
import load 
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)

model = open('C:/Users/patel/Google Drive/Colab Notebooks/model_ridge_class.sav', 'rb')
clf = joblib.load(model)

@app.route('/',methods = ['POST','GET'])
def index():
    
    vectorizer = joblib.load(open('C:/Users/patel/Google Drive/Colab Notebooks/vectorizer.sav'))
    transformer = joblib.load(open('C:/Users/patel/Google Drive/Colab Notebooks/transformer.sav'))

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        text = vectorizer.transform(data)
        text = transformer.fit_transform(text)
        my_prediction = clf.predict(text)
        my_prediction = np.round(my_prediction)
        my_prediction=str(my_prediction).strip('[]')

    return render_template('home.html',prediction = my_prediction)

if __name__ == "__main__":
    app.run()'''
