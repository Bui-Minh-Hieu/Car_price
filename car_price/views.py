from django.shortcuts import render
from .forms import CarPricePredictionForm
import joblib
import pandas as pd
import numpy as np
import os
import onnxruntime as ort

# Đường dẫn tải các file .joblib (Cần kiểm tra lại đường dẫn này)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sử dụng os.path.join cho từng thư mục con để xây dựng đường dẫn chính xác
OE_PATH = os.path.join(BASE_DIR, 'ml_model', 'car_price_oe.joblib')
OHE_PATH = os.path.join(BASE_DIR, 'ml_model', 'car_price_ohe.joblib')
TE_PATH = os.path.join(BASE_DIR, 'ml_model', 'car_price_te.joblib')
MODEL_PATH = os.path.join(BASE_DIR, 'ml_model', 'car_price_rf.onnx') 
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, 'ml_model', 'car_price_full_feature_names.joblib')

try:
    LOADED_MODEL = ort.InferenceSession(MODEL_PATH)
    LOADED_OE = joblib.load(OE_PATH)
    LOADED_OHE = joblib.load(OHE_PATH)
    LOADED_TE = joblib.load(TE_PATH)
    FEATURE_COLUMNS = joblib.load(FEATURE_COLUMNS_PATH)
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file mô hình/encoder. Đảm bảo file .joblib đã được lưu và đặt đúng thư mục.")
    LOADED_MODEL = None
except Exception as e:
    print(f"Lỗi khi tải hoặc chạy mô hình ONNX: {e}")
    LOADED_MODEL = None
    
def predict_car_price(data):
    if LOADED_MODEL is None:
        return 0
    input_data = {
        'Company Name': [data['hang_xe']],
        'Model Name': [data['model_xe']],
        'Model Year': [data['nam_san_xuat']],
        'Mileage': [np.log10(data['so_km'])],
        'Engine Type': [data['loai_dong_co']],
        'Engine Capacity': [data['dung_tich_dong_co']],
        'Color': [data['mau_xe']],           
        'Assembly': [data['day_chuyen_lap_rap']], 
        'Body Type': [data['kieu_dang_than_xe']], 
        'Transmission Type': [data['loai_hop_so']],
        'Registration Status': [data['dang_ky_xe']], 
    }
    
    df_new = pd.DataFrame(input_data)
    df_new.index = [0]

    # Các bộ mã hóa
    oe_cols = ['Assembly', 'Transmission Type', 'Registration Status']
    ohe_cols = ['Engine Type', 'Body Type']
    te_cols = ['Company Name', 'Model Name', 'Color']
    
    df_new[oe_cols] = LOADED_OE.transform(df_new[oe_cols])
    ohe_transformed = LOADED_OHE.transform(df_new[ohe_cols])
    ohe_feature_names = LOADED_OHE.get_feature_names_out()
    ohe_df = pd.DataFrame(ohe_transformed, columns=ohe_feature_names, index=df_new.index)
    df_new = pd.concat([df_new.drop(columns=ohe_cols), ohe_df], axis=1)
    df_new = LOADED_TE.transform(df_new)

    try:
        df_final = df_new[FEATURE_COLUMNS]
    except KeyError as e:
        # Nếu thiếu hoặc thừa cột, sẽ báo lỗi
        print(f"Lỗi Key: Không tìm thấy cột {e} trong DataFrame sau khi mã hóa. Kiểm tra lại tên cột/mã hóa.")
        return 0 # Trả về 0 nếu lỗi
    
    onnx_input_name = LOADED_MODEL.get_inputs()[0].name
    input_array = df_final.values.astype(np.float32)

    onnx_result = LOADED_MODEL.run(None, {onnx_input_name: input_array})
    
    # onnx_result[0] là mảng NumPy chứa kết quả dự đoán logarit
    log_price_pred_array = onnx_result[0] 
    
    # Đảo ngược logarit. Kết quả vẫn là mảng NumPy 1 phần tử
    final_price_array = np.power(10, log_price_pred_array) 
    
    # BƯỚC QUAN TRỌNG NHẤT: Trích xuất giá trị float đơn lẻ
    final_price_value = final_price_array[0][0] 
    
    return final_price_value # Trả về giá trị float đơn lẻ


def car_price_predictor_view(request):
    predicted_price = 0
    form = CarPricePredictionForm()

    if request.method == 'POST':
        form = CarPricePredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            predicted_price = predict_car_price(data)
            
    context = {
        'form': form,
        'predicted_price': round(predicted_price*0.0036 , 3) ,
    }
    
    return render(request, 'car_price/car_price_form.html', context)
    