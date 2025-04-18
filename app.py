import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import base64

# ✅ 兼容打包的资源路径函数
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ✅ 添加背景图片
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ 设置背景图路径并加载
bg_path = resource_path(os.path.join("Datasets", "Background.png"))
add_bg_from_local(bg_path)

# ✅ 按钮样式（清新绿色主题）
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-size: 18px;
    padding: 12px 30px;
    border-radius: 10px;
    border: none;
    box-shadow: 2px 4px 10px rgba(0,0,0,0.2);
    transition: 0.3s ease-in-out;
}
div.stButton > button:first-child:hover {
    background-color: #45a049;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# 页面标题
st.title("🌿 混凝土强度预测模型")

# ✅ 加载 CSV 数据
csv_path = resource_path(os.path.join("Datasets", "Data2.csv"))
try:
    df_clean = pd.read_csv(csv_path)
    st.success("初始化成功！")
except Exception as e:
    st.error(f"加载数据失败：{e}")
    st.stop()

# ✅ 训练模型
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

# ✅ UI 输入部分
st.markdown("### ✏️ 输入混凝土配合比参数")
col1, col2, col3, col4 = st.columns(4)
Cement_input = col1.number_input('Cement (kg/m³)', value=0.0, format="%.2f")
BFS_input = col2.number_input('Blast Furnace Slag (kg/m³)', value=0.0, format="%.2f")
Fly_Ash_input = col3.number_input('Fly Ash (kg/m³)', value=0.0, format="%.2f")
Water_input = col4.number_input('Water (kg/m³)', value=0.0, format="%.2f")

col5, col6, col7, col8 = st.columns(4)
SP_input = col5.number_input('Superplasticizer (kg/m³)', value=0.0, format="%.2f")
CA_input = col6.number_input('Coarse Aggregate (kg/m³)', value=0.0, format="%.2f")
FA_input = col7.number_input('Fine Aggregate (kg/m³)', value=0.0, format="%.2f")
Age_input = col8.number_input('Age (day)', value=0.0, format="%.2f")

user_input = pd.DataFrame([[Cement_input, BFS_input, Fly_Ash_input, Water_input,
                            SP_input, CA_input, FA_input, Age_input]], columns=features)

# ✅ 预测按钮与结果展示
if st.button("🌞 点击预测抗压强度"):
    user_input_scaled = ss_X.transform(user_input)
    user_prediction_scaled = model.predict(user_input_scaled).reshape(-1, 1)
    user_prediction = ss_y.inverse_transform(user_prediction_scaled)

    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #dfffe2, #a0f1b4, #c9ffd6);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            text-align: center;
            color: #003f2d;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: 20px;">
            <h2 style="margin-bottom: 15px;">🌼 预测结果</h2>
            <p style="font-size: 28px; font-weight: bold;">
                预测的抗压强度为：<span style="color: #1a7f37;">{user_prediction[0][0]:.2f} MPa</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
