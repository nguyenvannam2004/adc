# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np
# import pandas as pd  
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Tải các mô hình đã lưu
# model0 = joblib.load('PLA.pkl')
# model1 = joblib.load('logistic_model.pkl')
# model2 = joblib.load('neuralnetwork_model.pkl')
# model3 = joblib.load('ensemble_model.pkl')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     selected_model = data.get('model')

#     if selected_model == 'model1':
#         model = model1
#     elif selected_model == 'model2':
#         model = model2
#     elif selected_model == 'voting_classifier':
#         model = model3
#     elif selected_model == 'model0':
#         model = model0
#     else:
#         return jsonify({"message": "Mô hình không hợp lệ"}), 400
    
#     try:
#         features = [
#             float(data['age']),
#             float(data['sex']),
#             float(data['cp']),
#             float(data['trestbps']),
#             float(data['chol']),
#             float(data['fbs']),
#             float(data['restecg']),
#             float(data['thalach']),
#             float(data['exang']),
#             float(data['oldpeak']),
#             float(data['slope']),
#             float(data['ca']),
#             float(data['thal'])
#         ]
#     except (KeyError, ValueError) as e:
#         return jsonify({"message": "Dữ liệu đầu vào không hợp lệ", "error": str(e)}), 400

#     print("Features:", features)

#     # Chuyển đổi thành DataFrame với tên cột
#     feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
#                      'restecg', 'thalach', 'exang', 'oldpeak', 
#                      'slope', 'ca', 'thal']
#     features_df = pd.DataFrame([features], columns=feature_names)

#     # Dự đoán bằng mô hình
#     prediction = model.predict(features_df)

#     result = "Có nguy cơ mắc bệnh tim\n\n\n\n Đừng buồn, bạn hãy chạy vào trong vườn hái một quả chanh.\nBổ đôi nó ra(nhớ là phải bổ ngang nha) xong vắt nó vào cốc, cho 2 thìa đường,500ml nước đun sôi để nguội\nCho thêm 2 viên đá nữa cho mát rồi khuấy đều lên sẽ thu đc dung dịch hay còn gọi là nước đường.\nUống nó! Nó sẽ ko giúp bạn hết bị bệnh tim đâu nhưng mà nươc đường thì rất ngọt, với lại biết đâu đó có thể sẽ là lần cuối cùng mà bạn đc uống nước đường thì sao =))))" if prediction[0] == 1 else "Không có nguy cơ mắc bệnh tim"
    
#     return jsonify({"message": result})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
import streamlit as st
import joblib
import pandas as pd
import requests

# Tải các mô hình đã lưu
model0 = joblib.load('PLA.pkl')
model1 = joblib.load('logistic_model.pkl')
model2 = joblib.load('neuralnetwork_model.pkl')
model3 = joblib.load('ensemble_model.pkl')

# Tạo tiêu đề cho ứng dụng
st.title("Dự Đoán Bệnh Tim")

# Tạo một dropdown để chọn mô hình
model_options = {
    'model0': model0,
    'model1': model1,
    'model2': model2,
    'voting_classifier': model3
}

selected_model = st.selectbox("Chọn mô hình:", list(model_options.keys()))

# Nhập dữ liệu đầu vào
age = st.number_input("Tuổi:")
sex = st.number_input("Giới tính (0 hoặc 1):")
cp = st.number_input("Chỉ số đau ngực (0-3):")
trestbps = st.number_input("Huyết áp nghỉ (mm Hg):")
chol = st.number_input("Cholesterol (mg/dl):")
fbs = st.number_input("Đường huyết lúc nhịn ăn (0 hoặc 1):")
restecg = st.number_input("Kết quả điện tâm đồ (0-2):")
thalach = st.number_input("Tần số tim tối đa:")
exang = st.number_input("Đau thắt ngực khi hoạt động (0 hoặc 1):")
oldpeak = st.number_input("Độ dốc ST (0.0 - 6.2):")
slope = st.number_input("Độ dốc của đỉnh ST (0-2):")
ca = st.number_input("Số mạch máu lớn (0-3):")
thal = st.number_input("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect):")

# Nút dự đoán
if st.button("Dự đoán"):
    features = [
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]
    
    # Chuyển đổi thành DataFrame với tên cột
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 
                     'slope', 'ca', 'thal']
    features_df = pd.DataFrame([features], columns=feature_names)

    # Dự đoán bằng mô hình đã chọn
    model = model_options[selected_model]
    prediction = model.predict(features_df)

    # Kết quả dự đoán
    result = "Có nguy cơ mắc bệnh tim" if prediction[0] == 1 else "Không có nguy cơ mắc bệnh tim"
    st.success(result)
