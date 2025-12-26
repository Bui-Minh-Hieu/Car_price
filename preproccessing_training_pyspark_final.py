import json
import math
import numpy as np
import os

# Pyspark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, log10, lit, when
# --- CẬP NHẬT IMPORT QUAN TRỌNG ---
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StringIndexerModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator

# ONNX imports
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# ==============================================================================
# 1. KHỞI TẠO VÀ LOAD DỮ LIỆU
# ==============================================================================
spark = SparkSession.builder \
    .appName("CarPrice_Train_Export_KFold") \
    .master("local[*]") \
    .getOrCreate()

print("Đang đọc dữ liệu...")
# Đọc dữ liệu (đảm bảo file csv nằm đúng đường dẫn)
df = spark.read.csv("data/Clean Data_pakwheels.csv", header=True, inferSchema=True)
df = df.drop("Location")

# Xóa trùng lặp
print(f"Số dòng gốc: {df.count()}")
df = df.dropDuplicates()
print(f"Số dòng sau khi drop duplicates: {df.count()}")

# Log transform giá
df = df.withColumn("Price", log10(col("Price")))

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
print("Đang xử lý Feature Engineering...")

# --- A. TARGET ENCODING (Company, Model, Color) ---
target_enc_cols = ['Company Name', 'Model Name', 'Color']
smoothing = 5.0
global_mean = df.select(mean('Price')).collect()[0][0]

encoding_maps = {
    "global_mean": global_mean,
    "target_encoding": {},
    "ordinal_encoding": {},
    "onehot_encoding": {} 
}

df_encoded = df
for c in target_enc_cols:
    # 1. Tính toán thống kê
    # Tạo df với 3 cột: cột c đã được nhóm
    # cột cat_mean chứa giá trị trung bình của Price theo giá trị cột c
    # cột cat_count chứa số lượng giá trị cột c xuất hiện trong dữ lieuej
    stats = df.groupBy(c).agg(
        mean("Price").alias("cat_mean"),
        count("Price").alias("cat_count")
    )
    
    # 2. Áp dụng công thức smoothing
    # .withColumn(name, value): Tạo cột mới cho df, name tên cột, value giá trị của cột
    stats = stats.withColumn(
        c + "_encoded",
        (col("cat_count") * col("cat_mean") + smoothing * global_mean) / (col("cat_count") + smoothing)
    )
    
    # 3. Join vào bảng gốc
    # Left join các cột c_encoded của stats vào dữ liệu gốc theo cột c
    df_encoded = df_encoded.join(stats.select(c, c + "_encoded"), on=c, how="left")
    df_encoded = df_encoded.fillna({c + "_encoded": global_mean})
    
    rows = stats.select(c, c + "_encoded").collect()
    encoding_maps["target_encoding"][c] = {row[0]: row[1] for row in rows}

# --- B. ORDINAL ENCODING ---
# PySpark StringIndexer gán 0 cho giá trị xuất hiện nhiều nhất.
# Ta cần lưu lại quy tắc map này để Web App dùng đúng số đó.
ordinal_cols = ['Assembly', 'Transmission Type', 'Registration Status']
ordinal_stages = []     # Mảng lưu các đối tượng indexer, dùng cho pipeline

# Tạo bộ Indexer cho từng cột
for c in ordinal_cols:
    indexer = StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
    ordinal_stages.append(indexer)

# --- C. ONE-HOT ENCODING ---
ohe_cols = ['Engine Type', 'Body Type']
ohe_stages = []

for c in ohe_cols:
    # Indexer trước để chuyển kiểu chữ thành số
    indexer = StringIndexer(inputCol=c, outputCol=c + "_idx_temp", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=c + "_idx_temp", outputCol=c + "_vec", dropLast=False)
    ohe_stages.append(indexer)
    ohe_stages.append(encoder)

# Chạy Pipeline biến đổi Encoding
pre_pipeline = Pipeline(stages=ordinal_stages + ohe_stages)
pre_model = pre_pipeline.fit(df_encoded)
df_final = pre_model.transform(df_encoded)

# --- TRÍCH XUẤT MAPPING TỪ PIPELINE CHO JSON ---
# Duyệt từng stages trong pipeline tiền xử lý
for stage in pre_model.stages:
    # Kiểm tra nếu stage đó là StringIndexerModel
    if isinstance(stage, StringIndexerModel):
        # Lấy ra tên cột input của stage đó (ex: 'Model Name') và labels của stage đó (ex: labels=['Toyota', 'Honda'], Toyoa = 0 vì index=0)
        input_c = stage.getInputCol()
        labels = stage.labels

        # Ex: mapping = {'Toyota': 0, 'Honda': 1}
        mapping = {label: float(idx) for idx, label in enumerate(labels)}
        
        if input_c in ordinal_cols:
            encoding_maps["ordinal_encoding"][input_c] = mapping
        elif input_c in ohe_cols:
            encoding_maps["onehot_encoding"][input_c] = mapping

# ==============================================================================
# 3. TRAIN RANDOM FOREST VỚI K-FOLD
# ==============================================================================
print("Chuẩn bị dữ liệu huấn luyện...")

feature_list = (
    ["Model Year", "Mileage", "Engine Capacity"] +    
    [c + "_idx" for c in ordinal_cols] +              
    [c + "_vec" for c in ohe_cols] +                  
    [c + "_encoded" for c in target_enc_cols]         
)

assembler = VectorAssembler(inputCols=feature_list, outputCol="features")
train_df = assembler.transform(df_final).select("features", "Price")

# Chia tập dữ liệu: 80% để Train (có K-Fold bên trong), 20% để Test độc lập
train_data, test_data = train_df.randomSplit([0.8, 0.2], seed=40)

# Định nghĩa Model
rf = RandomForestRegressor(featuresCol="features", labelCol="Price", 
                           numTrees=200, maxDepth=12, seed=42)

# --- THIẾT LẬP K-FOLD ---
# ParamGrid
paramGrid = ParamGridBuilder().build()

# Evaluator: Dùng RMSE để đánh giá
evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")

# rossValidator: 10 Folds
cv = CrossValidator(estimator=rf,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=10, 
                    seed=42)

print("Đang chạy K-Fold Cross Validation (10 Folds)... Quá trình này có thể mất vài phút.")
cvModel = cv.fit(train_data)

# Lấy ra model tốt nhất
best_rf_model = cvModel.bestModel

# In kết quả đánh giá
print("-" * 30)
print("KẾT QUẢ K-FOLD:")
print(f"RMSE trung bình qua 10 folds: {min(cvModel.avgMetrics):.4f}")

# Đánh giá trên tập Test độc lập (Holdout set)
predictions = best_rf_model.transform(test_data)
rmse_test = evaluator.evaluate(predictions)
print(f"RMSE trên tập Test độc lập: {rmse_test:.4f}")
print("-" * 30)

# ==============================================================================
# 4. XUẤT MODEL RA FILE (JSON + ONNX)
# ==============================================================================

# Lưu file JSON
with open("car_price_mappings.json", "w", encoding='utf-8') as f:
    json.dump(encoding_maps, f, ensure_ascii=False, indent=4)
print("- Đã lưu mapping: car_price_mappings.json")

# Lưu model Random Forest ra ONNX
num_features = best_rf_model.numFeatures
print(f"Tổng số features model cần: {num_features}")

initial_types = [('features', FloatTensorType([None, num_features]))]

# Convert
print("Đang convert sang ONNX...")
onnx_model = onnxmltools.convert_sparkml(best_rf_model, initial_types=initial_types)

# Save
with open("car_price_rf.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("- Đã lưu model: car_price_rf.onnx")

spark.stop()
print("Hoàn tất!")