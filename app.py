# importing libraries
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from werkzeug.utils import secure_filename
import os
import cv2

# configuring flask
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

METRICS = [
      BinaryAccuracy(name='acc'),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc')
]

#loading models
tuberculosis_model = load_model('models/tuberculosisModel.h5')
covid_model = load_model('models/covidModel.h5')
pneumonia_model = load_model('models/pnuemoniaModel.h5')

cancer_model = load_model('models/cancerModel.h5')
cancer_model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=METRICS)

# preprocess the image
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

######################################## Routing Functions ########################################

@app.route('/')
def home():
    return render_template('mainpage.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/tuberculosis')
def tuberculosis():
    return render_template('tuberculosis.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/cancer')
def cancer():
    return render_template('cancer.html')


######################################## Prediction Functions ########################################

@app.route('/resultCovid', methods=['POST'])
def resultCovid():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            img_array = preprocess_image(img_path)

            # make predictions
            pred = covid_model.predict(img_array)

            if pred < 0.5:
                disease = "You are suffering with Covid"
                heading = "Treatment Recommendations"
                title1 = "Symptomatic Treatment:"
                desc1 = "Most cases of COVID-19 are mild and can be managed with supportive care at home. This includes getting plenty of rest, staying hydrated, and using over-the-counter medications to relieve symptoms such as fever, cough, and body aches. Acetaminophen (paracetamol) is typically recommended for fever and pain, while cough suppressants and expectorants may help with cough."
                title2 = "Hospitalization:"
                desc2 = "Patients with severe COVID-19 may require hospitalization, particularly if they experience difficulty breathing, persistent chest pain, confusion, or bluish lips or face. Hospitalized patients may receive supplemental oxygen therapy, intravenous fluids, and other supportive measures as needed."
                title3 = "Antiviral Therapy:"
                desc3 = "Several antiviral medications have been evaluated for the treatment of COVID-19, including remdesivir and molnupiravir. These medications may be considered for use in hospitalized patients with severe COVID-19, particularly those requiring supplemental oxygen. However, their efficacy varies, and treatment decisions should be based on individual patient factors and available clinical evidence."
            else:
                disease = "You are not suffering with Covid"
                heading = ""
                title1 = ""
                desc1 = ""
                title2 = ""
                desc2 = ""
                title3 = ""
                desc3 = ""
            return render_template('result_covid.html', prediction = pred, result = disease, heading = heading, title1 = title1, desc1 = desc1, title2 = title2, desc2 = desc2, title3 = title3, desc3 = desc3)

@app.route('/resultTuberculosis', methods=['POST'])
def resultTuberculosis():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            img_array = preprocess_image(img_path)

            # make predictions
            pred = tuberculosis_model.predict(img_array)

            if pred < 0.5:
                disease = "You are not suffering with Tuberculosis"
                heading = "Treatment Recommendation"
                title1 = "Infection Control:"
                desc1 = "Patients with active TB should be counseled on infection control measures to prevent the spread of the disease to others, including proper cough etiquette, ventilation of living spaces, and avoiding close contact with high-risk individuals."
                title2 = "Nutritional Support:"
                desc2 = "Adequate nutrition is important for patients with TB to support the immune system and aid in recovery. Nutritional counseling and supplementation may be necessary, especially for patients with malnutrition or HIV co-infection."
                title3 = "Drug Therapy:"
                desc3 = "TB is treated with a combination of antibiotics to prevent the development of drug resistance. The most commonly used drugs include: Isoniazid (INH), Rifampin (RIF), Pyrazinamide (PZA), Ethambutol (EMB)"
            else:
                disease = "You are suffering with Tuberculosis"
                heading = ""
                title1 = ""
                desc1 = ""
                title2 = ""
                desc2 = ""
                title3 = ""
                desc3 = ""
            if pred < 0.5:
                outcome = 0
            else:
                outcome = 1

            return render_template('result_tuberculosis.html', filename=filename, result=disease,  heading = heading, title1 = title1, desc1 = desc1, title2 = title2, desc2 = desc2, title3 = title3, desc3 = desc3)

@app.route('/resultPneumonia', methods=['POST'])
def resultPneumonia():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            img_array = preprocess_image(img_path)

            # make predictions
            pred = pneumonia_model.predict(img_array)

            if pred < 0.5:
                disease = "You are not suffering with Pneumonia"
                heading = "Treatment Recommendations"
                title1 = "Antibiotics:"
                desc1 = "If the pneumonia is bacterial, antibiotics are typically prescribed. The choice of antibiotic depends on factors such as the suspected organism and local antibiotic resistance patterns. Commonly used antibiotics include amoxicillin, azithromycin, clarithromycin, or levofloxacin. For severe cases or in patients with risk factors for drug-resistant bacteria, broad-spectrum antibiotics like ceftriaxone or a fluoroquinolone may be used."
                title2 = "Hospitalization:"
                desc2 = "Severe cases of pneumonia, especially in older adults or those with underlying health conditions, may require hospitalization. This allows for close monitoring, intravenous antibiotics, and oxygen therapy if needed."
                title3 = "Corticosteroids:"
                desc3 = "In some cases, especially for severe pneumonia or when there is significant inflammation, corticosteroids may be used as adjunctive therapy to reduce inflammation and improve oxygenation."
            else:
                disease = "You are suffering with Pneumonia"
                heading = ""
                title1 = ""
                desc1 = ""
                title2 = ""
                desc2 = ""
                title3 = ""
                desc3 = ""
            return render_template('result_pneumonia.html', filename=filename, result=disease, heading = heading, title1 = title1, desc1 = desc1, title2 = title2, desc2 = desc2, title3 = title3, desc3 = desc3)

@app.route('/resultCancer', methods=['POST'])
def resultCancer():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            img_array = preprocess_image(img_path)

            # make predictions
            pred = cancer_model.predict(img_array)
            predicted_class = np.argmax(pred)

            if predicted_class == 0:
                disease = "You are suffering with Adenocarcinoma"
                heading = "Treatment Recommendations"
                title1 = "Surgery:"
                desc1 = "Surgical removal of the tumor is often the primary treatment for localized adenocarcinoma, especially if the tumor is small and hasn't spread to other organs. The goal of surgery is to remove as much of the tumor as possible while preserving healthy surrounding tissue. In some cases, lymph nodes may also be removed to assess for spread of the cancer."
                title2 = "Chemotherapy:"
                desc2 = "Chemotherapy may be used before or after surgery (neoadjuvant or adjuvant therapy) to shrink the tumor, kill remaining cancer cells, or reduce the risk of recurrence. Chemotherapy drugs can be administered orally or intravenously and may be used alone or in combination with other treatments."
                title3 = "Radiation Therapy:"
                desc3 = "Radiation therapy uses high-energy beams to kill cancer cells or shrink tumors. It may be used as a primary treatment for localized adenocarcinoma, especially when surgery is not an option, or it may be used after surgery to kill any remaining cancer cells."
            if predicted_class == 1:
                disease = "You are suffering with Cell Carcinoma"
                heading = "Treatment Recommendations"
                title1 = "Surgery:"
                desc1 = "Surgical removal of the tumor is often the primary treatment for localized carcinoma. The goal of surgery is to remove the tumor along with a margin of healthy tissue to ensure complete removal of cancerous cells. In some cases, lymph nodes may also be removed to check for the spread of cancer."
                title2 = "Radiation Therapy:"
                desc2 = "Radiation therapy uses high-energy rays to kill cancer cells or shrink tumors. It may be used as a primary treatment for localized carcinoma, particularly when surgery is not feasible or as an adjuvant therapy after surgery to destroy any remaining cancer cells."
                title3 = "Chemotherapy:"
                desc3 = "Chemotherapy involves the use of drugs to kill cancer cells or slow their growth. It may be used alone or in combination with other treatments such as surgery or radiation therapy. Chemotherapy is often recommended for carcinomas that have spread beyond the primary site (metastatic carcinoma)."
            if predicted_class == 2:
                disease = "You are not suffering with Cancer"
                heading = ""
                title1 = ""
                desc1 = ""
                title2 = ""
                desc2 = ""
                title3 = ""
                desc3 = ""
            return render_template('result_cancer.html', filename=filename, result=disease, heading = heading, title1 = title1, desc1 = desc1, title2 = title2, desc2 = desc2, title3 = title3, desc3 = desc3)

if __name__ == '__main__':
    app.run(debug=True)
