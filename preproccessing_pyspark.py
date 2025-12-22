import json
import math
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, log10, lit, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

# Hỗ trợ xuất ONNX
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# ==============================================================================
# 1. KHỞI TẠO VÀ LOAD DỮ LIỆU
# ==============================================================================
spark = SparkSession.builder \
    .appName("CarPrice_Train_Export") \
    .master("local[*]") \
    .getOrCreate()

# Đọc dữ liệu, tự động nhận diện kiểu số
df = spark.read.csv("data/Clean Data_pakwheels.csv", header=True, inferSchema=True)
df = df.drop("Location")

# Xóa trùng lặp (Giống cell 4-5)
print(f"Số dòng gốc: {df.count()}")
df = df.dropDuplicates()
print(f"Số dòng sau khi drop duplicates: {df.count()}")

# Log transform giá (Giống cell 9)
df = df.withColumn("Price", log10(col("Price")))

# ==============================================================================
# 2. FEATURE ENGINEERING (MÔ PHỎNG LẠI SKLEARN)
# ==============================================================================

# --- A. TARGET ENCODING (Company, Model, Color) ---
# Logic: Tính trên toàn bộ dataset trước khi split (Giống cell 28 file gốc)
target_enc_cols = ['Company Name', 'Model Name', 'Color']
smoothing = 5.0
global_mean = df.select(mean('Price')).collect()[0][0]

# Dictionary để lưu mapping cho Web App
encoding_maps = {
    "global_mean": global_mean,
    "target_encoding": {},
    "ordinal_encoding": {},
    "onehot_encoding": {} # Chỉ lưu danh sách cột để biết thứ tự
}

df_encoded = df
for c in target_enc_cols:
    # 1. Tính toán thống kê
    stats = df.groupBy(c).agg(
        mean("Price").alias("cat_mean"),
        count("Price").alias("cat_count")
    )
    
    # 2. Áp dụng công thức smoothing
    stats = stats.withColumn(
        c + "_encoded",
        (col("cat_count") * col("cat_mean") + smoothing * global_mean) / (col("cat_count") + smoothing)
    )
    
    # 3. Join vào bảng gốc
    df_encoded = df_encoded.join(stats.select(c, c + "_encoded"), on=c, how="left")
    df_encoded = df_encoded.fillna({c + "_encoded": global_mean})
    
    # 4. Lưu mapping vào dict để xuất JSON
    rows = stats.select(c, c + "_encoded").collect()
    encoding_maps["target_encoding"][c] = {row[0]: row[1] for row in rows}

# --- B. ORDINAL ENCODING (Assembly, Transmission, Registration) ---
# PySpark StringIndexer gán 0 cho giá trị xuất hiện nhiều nhất.
# Ta cần lưu lại quy tắc map này để Web App dùng đúng số đó.
ordinal_cols = ['Assembly', 'Transmission Type', 'Registration Status']
ordinal_stages = []

for c in ordinal_cols:
    indexer = StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    ordinal_stages.append(indexer)

# --- C. ONE-HOT ENCODING (Engine Type, Body Type) ---
ohe_cols = ['Engine Type', 'Body Type']
ohe_stages = []

for c in ohe_cols:
    # Indexer trước
    indexer = StringIndexer(inputCol=c, outputCol=c + "_idx_temp", handleInvalid="keep")
    # Encoder sau (dropLast=False để giống sklearn sparse=False)
    encoder = OneHotEncoder(inputCol=c + "_idx_temp", outputCol=c + "_vec", dropLast=False)
    ohe_stages.append(indexer)
    ohe_stages.append(encoder)

# Chạy Pipeline biến đổi Encoding
pre_pipeline = Pipeline(stages=ordinal_stages + ohe_stages)
pre_model = pre_pipeline.fit(df_encoded)
df_final = pre_model.transform(df_encoded)

# --- TRÍCH XUẤT MAPPING TỪ PIPELINE CHO JSON ---
# Phần này cực quan trọng: Lấy ra quy tắc mà Spark đã gán số cho chữ
for stage in pre_model.stages:
    if isinstance(stage,  from pyspark.ml.feature import StringIndexerModel):
        # Kiểm tra xem đây là indexer cho Ordinal hay OHE
        input_c = stage.getInputCol()
        labels = stage.labels
        # Tạo dict: {'Imported': 0.0, 'Local': 1.0, ...}
        mapping = {label: float(idx) for idx, label in enumerate(labels)}
        
        if input_c in ordinal_cols:
            encoding_maps["ordinal_encoding"][input_c] = mapping
        elif input_c in ohe_cols:
            # Với OHE, ta cần biết index nào ứng với giá trị nào để tạo vector
            encoding_maps["onehot_encoding"][input_c] = mapping

# ==============================================================================
# 3. TRAIN RANDOM FOREST
# ==============================================================================

# Gom features
# Lưu ý thứ tự này phải KHỚP TUYỆT ĐỐI với lúc tạo array ở Web App
feature_list = (
    ["Model Year", "Mileage", "Engine Capacity"] +    # Numeric gốc
    [c + "_idx" for c in ordinal_cols] +              # Ordinal Encoded
    [c + "_vec" for c in ohe_cols] +                  # OneHot Encoded (Vectors)
    [c + "_encoded" for c in target_enc_cols]         # Target Encoded
)

assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
train_df = assembler.transform(df_final).select("features", "Price")

# Split Data (Giống cell 22 file gốc dùng random_state=40)
train_data, test_data = train_df.randomSplit([0.8, 0.2], seed=40)

# Train Model (Giống cell 22 file gốc dùng random_state=42)
rf = RandomForestRegressor(featuresCol="features", labelCol="Price", 
                           numTrees=200, maxDepth=12, seed=42)
rf_model = rf.fit(train_data)

print("Đã huấn luyện xong!")

# ==============================================================================
# 4. XUẤT MODEL RA FILE (JSON + ONNX)
# ==============================================================================

# 4.1 Lưu file JSON chứa các quy tắc mapping (TargetEnc, Ordinal, OHE)
with open("car_price_mappings.json", "w", encoding='utf-8') as f:
    json.dump(encoding_maps, f, ensure_ascii=False, indent=4)
print("- Đã lưu mapping: car_price_mappings.json")

# 4.2 Lưu model Random Forest ra ONNX
# Tính tổng số features đầu vào để khai báo cho ONNX
# Numeric(3) + Ordinal(3) + TargetEnc(3) + OHE Vectors (phải tính len)
num_features = rf_model.numFeatures
print(f"Tổng số features model cần: {num_features}")

initial_types = [('features', FloatTensorType([None, num_features]))]

# Convert
onnx_model = onnxmltools.convert_sparkml(rf_model, initial_types=initial_types)

# Save
with open("car_price_rf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("- Đã lưu model: car_price_rf.onnx")

# Dừng Spark
spark.stop()