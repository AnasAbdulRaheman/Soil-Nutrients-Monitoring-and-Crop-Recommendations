import string
import bcrypt
from flask import Flask, redirect, render_template, url_for, request, Markup, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from utils.fertilizer import fertilizer_dic
from utils.disease import disease_dic
from utils.ml_models import load_random_forest_model

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = 'thisissecretkey'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

latest_ph = None
ph_value = None
temp_value = None
humidity_value = None

# Load Models
crop_recommendation_model = load_random_forest_model()
if crop_recommendation_model is None:
    raise Exception("❌ RandomForest model could not be loaded.")

disease_classes = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
    'Blueberry__healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)_healthy',
    'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust',
    'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape__Black_rot',
    'Grape_Esca(Black_Measles)', 'GrapeLeaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy',
    'Orange_Haunglongbing(Citrus_greening)', 'Peach_Bacterial_spot', 'Peach__healthy',
    'Pepper,bell_Bacterial_spot', 'Pepper,bellhealthy', 'Potato__Early_blight',
    'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy',
    'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
    'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']  # same long list
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# DB Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class UserAdmin(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class ContactUs(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(500), nullable=False)
    text = db.Column(db.String(900), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nitrogen = db.Column(db.Float, nullable=False)
    phosphorous = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Forms
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)])
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)])
    submit = SubmitField("Register")

    def validate_username(self, username):
        if User.query.filter_by(username=username.data).first():
            raise ValidationError("Username already exists.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)])
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)])
    submit = SubmitField("Login")

# Utils
def weather_fetch(city_name):
    api_key = config.weather_api_key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"
    x = requests.get(url).json()
    if x["cod"] != "404":
        y = x["main"]
        return round((y["temp"] - 273.15), 2), y["humidity"]
    return None

def predict_image(img, model=disease_model):
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# Routes
@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        entry = ContactUs(
            name=request.form['name'],
            email=request.form['email'],
            text=request.form['text']
        )
        db.session.add(entry)
        db.session.commit()
    return render_template("contact.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template("login.html", form=form)

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, password=hashed_pw)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("signup.html", form=form)

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('hello_world'))

@app.route("/crop-recommend", methods=['GET', 'POST'])
@login_required
def crop_recommend():
    global ph_value, temp_value, humidity_value
    prediction = None
    latest = SensorData.query.order_by(SensorData.timestamp.desc()).first()

    if request.method == 'POST':
        try:
            data = np.array([[float(request.form['nitrogen']),
                              float(request.form['phosphorous']),
                              float(request.form['potassium']),
                              temp_value or float(request.form['temperature']),
                              humidity_value or float(request.form['humidity']),
                              ph_value or float(request.form['ph']),
                              float(request.form['rainfall'])]])
            prediction = crop_recommendation_model.predict(data)[0]
        except Exception as e:
            prediction = f"Invalid input: {e}"

    return render_template("crop.html", ph_val=ph_value, data=latest, prediction=prediction)

@app.route("/crop-predict", methods=['POST'])
@login_required
def crop_prediction():
    global ph_value, temp_value, humidity_value
    try:
        latest = SensorData.query.order_by(SensorData.timestamp.desc()).first()

        temp_value = latest.temperature if latest else float(request.form['temperature'])
        humidity_value = latest.humidity if latest else float(request.form['humidity'])
        ph_value = latest.ph if latest else float(request.form['ph'])

        data = np.array([[float(request.form['nitrogen']),
                          float(request.form['phosphorous']),
                          float(request.form['potassium']),
                          temp_value,
                          humidity_value,
                          ph_value,
                          float(request.form['rainfall'])]])

        prediction = crop_recommendation_model.predict(data)[0]

        return render_template("crop-result.html", prediction=prediction)
    except Exception as e:
        return render_template("crop-result.html", prediction=f"❌ Error: {e}")

@app.route("/fertilizer")
@login_required
def fertilizer_recommendation():
    return render_template("fertilizer.html")

@app.route("/fertilizer-predict", methods=['POST'])
@login_required
def fert_recommend():
    crop = request.form['cropname']
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['potassium'])

    df = pd.read_csv('Data/fertilizer.csv')
    nr, pr, kr = df[df['Crop'] == crop][['N', 'P', 'K']].iloc[0]
    diff = {'N': nr - N, 'P': pr - P, 'K': kr - K}
    #key = max(diff, key=lambda x: abs(diff[x])) + ("High" if diff[max(diff, key=abs)] < 0 else "low")
    max_key = max(diff, key=lambda x: abs(diff[x]))
    key = max_key + ("High" if diff[max_key] < 0 else "low")

    return render_template('fertilizer-result.html', recommendation=Markup(fertilizer_dic[key]))

@app.route("/disease-predict", methods=['GET', 'POST'])
@login_required
def disease_prediction():
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file:
            raw_prediction = predict_image(file.read())
            print("Raw prediction:", raw_prediction)

            # Attempt to match format to keys in disease_dic
            fixed_prediction = raw_prediction.replace('_', '___', 1)
            print("Fixed prediction:", fixed_prediction)

            # Check if either fixed or raw prediction exists
            disease_info = disease_dic.get(fixed_prediction) or disease_dic.get(raw_prediction)

            if disease_info:
                return render_template('disease-result.html', prediction=Markup(disease_info))
            else:
                return render_template('disease-result.html',
                                       prediction="❗ Disease info not found for prediction: " + raw_prediction)

    return render_template("disease.html")


 
@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('admindashboard'))
    if form.validate_on_submit():
        user = UserAdmin.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('admindashboard'))
    return render_template("adminlogin.html", form=form)

@app.route("/admindashboard")
@login_required
def admindashboard():
    return render_template("admindashboard.html", alltodo=ContactUs.query.all(), alluser=User.query.all())

@app.route("/reg", methods=['GET', 'POST'])
def reg():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_pw = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        admin = UserAdmin(username=form.username.data, password=hashed_pw)
        db.session.add(admin)
        db.session.commit()
        return redirect(url_for('AdminLogin'))
    return render_template("reg.html", form=form)

@app.route("/ph-update", methods=['POST'])
def update_ph():
    global latest_ph, ph_value
    try:
        data = request.get_json(force=True)
        if data and 'ph' in data:
            latest_ph = data['ph']
            ph_value = latest_ph
            return jsonify({'status': 'success', 'ph': latest_ph}), 200
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update-sensor', methods=['POST'])
def update_sensor():
    try:
        data = request.get_json()
        npk = data.get("npk", [0, 0, 0])
        ph = data.get("ph", 0)
        temperature = data.get("temperature", 0)
        humidity = data.get("humidity", 0)

        if len(npk) == 3:
            new_data = SensorData(
                nitrogen=npk[0],
                phosphorous=npk[1],
                potassium=npk[2],
                ph=ph,
                temperature=temperature,
                humidity=humidity,
                rainfall=0
            )
            db.session.add(new_data)
            db.session.commit()
            return jsonify({"message": "Data saved successfully"}), 200
        else:
            return jsonify({"error": "Invalid NPK data"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ph-display')
def display_ph():
    return render_template('ph_display.html', ph=latest_ph)

@app.route('/get-latest-ph')
def get_latest_ph():
    return jsonify({'ph': ph_value})

@app.route('/latest-sensor-json', methods=['GET'])
def latest_sensor_json():
    try:
        latest = SensorData.query.order_by(SensorData.timestamp.desc()).first()
        if latest:
            return jsonify({
                "npk": [latest.nitrogen, latest.phosphorous, latest.potassium],
                "ph": latest.ph,
                "temperature": latest.temperature,
                "humidity": latest.humidity,
                "rainfall": latest.rainfall,
                "timestamp": latest.timestamp
            })
        else:
            return jsonify({"error": "No data found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/temp-humidity-update", methods=['POST'])
def update_temp_humidity():
    global temp_value, humidity_value
    try:
        data = request.get_json(force=True)
        temp = data.get("temperature")
        humidity = data.get("humidity")
        if temp is not None and humidity is not None:
            temp_value = temp
            humidity_value = humidity
            return jsonify({'status': 'success', 'temperature': temp, 'humidity': humidity}), 200
        return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# MAIN
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=8000, debug=True)