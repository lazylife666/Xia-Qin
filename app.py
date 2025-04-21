import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import base64

# 路径转换助手
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 加载背景图
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

# 全局字体设置和居中
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Times New Roman", "KaiTi", serif !important;
    font-size: 22px !important;
    line-height: 2;
    text-align: center;
}

h1 {
    font-size: 72px !important;
    font-weight: 900 !important;
    color: #6a4c2f !important;
    letter-spacing: 2px;
    margin: 60px auto 40px auto !important;
    font-family: "Times New Roman", "KaiTi", serif !important;
    white-space: nowrap;
    text-align: center;
}

h2 {
    font-size: 32px !important;
    font-weight: bold;
    color: #a0522d;
    text-align: center;
    margin-bottom: 10px;
    font-family: "Times New Roman", "KaiTi", serif !important;
}

h3, h4, h5, h6, p, span, div, label {
    font-family: "Times New Roman", "KaiTi", serif !important;
    font-size: 24px !important;
    line-height: 2 !important;
    text-align: right;
}

input, textarea, button {
    font-family: "Times New Roman", "KaiTi", serif !important;
    font-size: 22px !important;
}

.stNumberInput label {
    font-size: 22px !important;
    text-align: center;
}

/* 美化 selectbox 样式 */
.css-1wa3eu0 {
    font-family: "Times New Roman", "KaiTi", serif !important;
    font-size: 22px !important;
    text-align: center !important;
}
.css-1ok3k2k {
    justify-content: center !important;
}

/* 调整预测按钮样式和大小 */
div.stButton > button:first-child {
    background: linear-gradient(to right, #d2b48c, #c0a070);
    color: white;
    font-size: 26px;
    font-weight: bold;
    padding: 28px 70px;
    border-radius: 16px;
    border: none;
    box-shadow: 0 4px 15px rgba(210, 180, 140, 0.4);
    transition: 0.3s ease-in-out;
    margin: 40px auto;
    display: block;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(to right, #c1a271, #a68a6d);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

data_folder = "数据集"
bg_path = resource_path(os.path.join(data_folder, "Background.png"))
add_bg_from_local(bg_path)

if "lang" not in st.session_state:
    st.session_state.lang = "中文"

lang = st.radio("语言 Language", ["中文", "English"], horizontal=True)
st.session_state.lang = lang

# 文本映射
txt = {
    "title": {
        "中文": "🧱 混凝土强度预测系统",
        "English": "🧱 Concrete Strength Predictor"
    },
    "section_input": {
        "中文": "✏️  输入混凝土配合比参数",
        "English": "✏️  Input Concrete Mix Parameters"
    },
    "button_predict": {
        "中文": "🌟 开始预测",
        "English": "🌟 Predict Now"
    },
    "result_header": {
        "中文": "📊 预测结果",
        "English": "📊 Prediction Result"
    },
    "result_text": {
        "中文": "预测的抗压强度为：",
        "English": "Predicted compressive strength:"
    },
    "features": [
        {"中文": "水泥 (kg/m³)", "English": "Cement (kg/m³)"},
        {"中文": "矿渣 (kg/m³)", "English": "Blast Furnace Slag (kg/m³)"},
        {"中文": "粉煤灰 (kg/m³)", "English": "Fly Ash (kg/m³)"},
        {"中文": "水 (kg/m³)", "English": "Water (kg/m³)"},
        {"中文": "减水剂 (kg/m³)", "English": "Superplasticizer (kg/m³)"},
        {"中文": "粗骨料 (kg/m³)", "English": "Coarse Aggregate (kg/m³)"},
        {"中文": "细骨料 (kg/m³)", "English": "Fine Aggregate (kg/m³)"},
        {"中文": "龄期 (day)", "English": "Age (day)"}
    ],
    "unit": "MPa"
}

st.markdown(f"""
<h1>{txt['title'][st.session_state.lang]}</h1>
""", unsafe_allow_html=True)

file_path = os.path.join(data_folder, "Data2.csv")
try:
    df_clean = pd.read_csv(file_path)
except Exception as e:
    st.error(f"加载数据失败：{e}")
    st.stop()

features = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
            'Coarse Aggregate', 'Fine Aggregate', 'Age']
X = df_clean[features]
y = df_clean[['Concrete compressive strength']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = ss_y.transform(y_test.values.reshape(-1, 1))

model = xgb.XGBRegressor(
    booster='gbtree',
    eval_metric='rmse',
    gamma=0.001,
    min_child_weight=1,
    max_depth=4,
    subsample=0.5,
    colsample_bytree=0.6,
    tree_method='exact',
    learning_rate=0.1,
    n_estimators=500,
    nthread=1,
    seed=200
)
model.fit(X_train, y_train)

# 输入参数整体右移
st.markdown(f"""
<div style="margin-left: 80px;">
<h2 style="text-align: left; font-size: 36px; font-weight: bold;">
    {txt['section_input'][st.session_state.lang]}
</h2>
""", unsafe_allow_html=True)

labels = txt['features']
columns = st.columns([1, 0.2, 1, 0.2])

# 上文省略，以下是参数输入部分

Cement_input = columns[0].number_input(txt['features'][0][lang], min_value=102, max_value=540, value=102, step=1)
BFS_input = columns[2].number_input(txt['features'][1][lang], min_value=0, max_value=359, value=0, step=1)

columns = st.columns([1, 0.2, 1, 0.2])
Fly_Ash_input = columns[0].number_input(txt['features'][2][lang], min_value=0, max_value=200, value=0, step=1)
Water_input = columns[2].number_input(txt['features'][3][lang], min_value=122, max_value=247, value=122, step=1)

columns = st.columns([1, 0.2, 1, 0.2])
SP_input = columns[0].number_input(txt['features'][4][lang], min_value=0, max_value=32, value=0, step=1)
CA_input = columns[2].number_input(txt['features'][5][lang], min_value=801, max_value=1145, value=801, step=1)

columns = st.columns([1, 0.2, 1, 0.2])
FA_input = columns[0].number_input(txt['features'][6][lang], min_value=594, max_value=992, value=594, step=1)
Age_input = columns[2].selectbox(txt['features'][7][lang], options=[7, 28, 56, 96], index=0)

user_input = pd.DataFrame([[Cement_input, BFS_input, Fly_Ash_input, Water_input,
                            SP_input, CA_input, FA_input, Age_input]], columns=features)

st.markdown("</div>", unsafe_allow_html=True)

user_input = pd.DataFrame([[Cement_input, BFS_input, Fly_Ash_input, Water_input,
                            SP_input, CA_input, FA_input, Age_input]], columns=features)

if st.button(txt['button_predict'][lang]):
    user_input_scaled = ss_X.transform(user_input)
    user_prediction_scaled = model.predict(user_input_scaled).reshape(-1, 1)
    user_prediction = ss_y.inverse_transform(user_prediction_scaled)

    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(to right, #e0cba3, #c8a97e);
            padding: 50px;
            border-radius: 22px;
            box-shadow: 0 4px 20px rgba(200, 160, 120, 0.3);
            text-align: center;
            color: #4d3a2d;
            font-family: 'Times New Roman', 'KaiTi', serif;
            margin-top: 30px;">
            <h2 style="margin-bottom: 25px; font-size: 36px;">{txt['result_header'][lang]}</h2>
            <p style="font-size: 34px; font-weight: bold;">
                {txt['result_text'][lang]}<span style="color: #a67b5b;">
                {user_prediction[0][0]:.2f} {txt['unit']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)