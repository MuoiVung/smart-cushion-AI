import pandas as pd
import numpy as np
import os
import glob
import joblib
import re
import shutil
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ========================================================
# 0. POSTURE CONFIGURATION
# ========================================================
POSTURE_INFO = {
    0: {"label": "NUP",  "name": "Neutral Upright Posture"},
    1: {"label": "LF",   "name": "Leaning Forward"},
    2: {"label": "LB",   "name": "Leaning Backward"},
    3: {"label": "LFSR", "name": "Leaning Forward & Support Right"},
    4: {"label": "LFSL", "name": "Leaning Forward & Support Left"},
    5: {"label": "CRL",  "name": "Cross Right Leg"},
    6: {"label": "CLL",  "name": "Cross Left Leg"},
    7: {"label": "CRLL", "name": "Cross Right Thigh"},
    8: {"label": "CLLL", "name": "Cross Left Thigh"}
}

FILE_MAP = {
    "straight": 0, "nup": 0,
    "leaning_forward": 1, "lf": 1,
    "leaning_backward": 2, "lb": 2,
    "leaning_forward_support_right": 3, "lfsr": 3,
    "leaning_forward_support_left": 4, "lfsl": 4,
    "cross_right_leg": 5, "crl": 5,
    "cross_left_leg": 6, "cll": 6,
    "cross_right_leg_legged": 7, "crll": 7,
    "cross_left_leg_legged": 8, "clll": 8
}

# ========================================================
# 1. DATA PROCESSING PIPELINE
# ========================================================

def prepare_data(data_folder='./lab_07052026'):
    print(f"--- LOADING DATA FROM {data_folder} ---")
    all_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
    
    if not all_files:
        print(f"❌ ERROR: No .xlsx files found in {data_folder}")
        return None, None, None

    np.random.seed(42)
    np.random.shuffle(all_files)
    
    fsr_cols = ['FSR Front Left', 'FSR Front Mid', 'FSR Front Right', 
                'FSR Mid Left', 'FSR Mid Mid', 'FSR Mid Right',
                'FSR Back Left', 'FSR Back Mid', 'FSR Back Right']

    X_list, y_list, group_list = [], [], []

    # Sắp xếp key theo độ dài giảm dần để ưu tiên khớp các tên dài/cụ thể trước
    # Ví dụ: khớp 'lfsl' trước khi khớp 'lf'
    sorted_keys = sorted(FILE_MAP.keys(), key=len, reverse=True)

    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path).lower()
        
        # Tìm nhãn dựa trên FILE_MAP
        found_label = None
        for key in sorted_keys:
            if key in filename:
                found_label = FILE_MAP[key]
                break
        
        if found_label is None:
            continue

        df = pd.read_excel(file_path)
        # Bỏ 5s đầu và 5s cuối để ổn định
        if len(df) > 100:
            df = df.iloc[50:-50]
        
        # Lấy 9 cảm biến
        X_file = df[fsr_cols].values
        y_file = np.full(len(X_file), found_label)
        groups = np.full(len(X_file), i) # ID file để chia GroupKFold
        
        X_list.append(X_file)
        y_list.append(y_file)
        group_list.append(groups)
        print(f"  + Loaded {len(X_file)} frames: {os.path.basename(file_path)} -> Class: {found_label}")

    if not X_list:
        print("❌ ERROR: No valid data found!")
        return None, None, None

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)
    groups_all = np.concatenate(group_list)
    
    # 1. Cơ bản: Normalize L1 (Relative Pressure)
    row_sums = X_all.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X_rel = X_all / row_sums

    # 2. Feature Engineering: Thêm các đặc trưng thông minh
    # [FL, FM, FR, ML, MM, MR, BL, BM, BR]
    # Indices: 0, 1, 2, 3, 4, 5, 6, 7, 8
    
    front_sum = X_rel[:, [0, 1, 2]].sum(axis=1, keepdims=True)
    back_sum = X_rel[:, [6, 7, 8]].sum(axis=1, keepdims=True)
    left_sum = X_rel[:, [0, 3, 6]].sum(axis=1, keepdims=True)
    right_sum = X_rel[:, [2, 5, 8]].sum(axis=1, keepdims=True)
    
    # Tỷ lệ Trước/Sau và Trái/Phải
    fb_ratio = (front_sum - back_sum) / (front_sum + back_sum + 1e-5)
    lr_ratio = (left_sum - right_sum) / (left_sum + right_sum + 1e-5)
    
    # Kết hợp lại thành bộ Feature mới
    X_final = np.hstack([X_rel, fb_ratio, lr_ratio])

    print(f"\n=> TOTAL VALID FRAMES: {len(X_final)}")
    print(f"=> FEATURES USED: 9 Raw + 2 Engineered = 11")
    return X_final, y_all, groups_all

# ========================================================
# 2. XGBOOST TRAINING (GROUP-BASED)
# ========================================================

def train_xgb_cv(X_all, y_all, groups_all):
    # Sử dụng LabelEncoder để đảm bảo nhãn liên tục từ 0 (Bắt buộc cho XGBoost)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)
    
    # Lưu bản đồ nhãn để in kết quả chính xác
    # mapped_classes[0] sẽ là nhãn gốc (ví dụ: 0), mapped_classes[1] là nhãn 1, v.v.
    mapped_classes = le.classes_
    
    print(f"--- STARTING 5-FOLD CV WITH XGBOOST ({len(mapped_classes)} classes) ---")
    skf = GroupKFold(n_splits=5)
    
    fold_no = 1
    acc_per_fold = []
    best_model, best_acc = None, 0.0

    for train_index, test_index in skf.split(X_all, y_encoded, groups=groups_all):
        X_train, X_test = X_all[train_index], X_all[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Khởi tạo mô hình XGBoost
        # - n_estimators: Số lượng vòng boost
        # - learning_rate: Tốc độ học (nhỏ giúp mô hình ổn định)
        # - max_depth: Độ sâu cây
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Dự đoán tập test
        y_pred = xgb_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"-> Fold {fold_no} COMPLETED | Accuracy: {acc:.2f}%")
        
        acc_per_fold.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_model = xgb_model
        
        fold_no += 1

    print("\n" + "="*50)
    print("SUMMARY OF XGBOOST RESULTS")
    print("="*50)
    print(f"Mean Accuracy: {np.mean(acc_per_fold):.2f}% (+/- {np.std(acc_per_fold):.2f}%)")
    
    # In tầm quan trọng của các cảm biến
    print("\n--- MỨC ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG ---")
    fsr_cols = ['FL', 'FM', 'FR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR']
    feature_names = fsr_cols + ["FB_Ratio", "LR_Ratio"]
    importances = best_model.feature_importances_
    
    for name, imp in zip(feature_names, importances):
        print(f"{name}: {imp*100:.1f}%")

    # --- TỰ ĐỘNG ĐÁNH VERSION VÀ ĐỒNG BỘ ---
    model_base = "posture_xgb_model"
    le_base = "posture_xgb_le"
    extension = ".pkl"
    
    # Tìm version tiếp theo dựa trên file model
    existing_models = glob.glob(f"{model_base}_v*{extension}")
    max_v = 0
    for f in existing_models:
        match = re.search(r'_v(\d+)', f)
        if match: max_v = max(max_v, int(match.group(1)))
    
    new_v = f"v{max_v + 1}"
    model_name = f"{model_base}_{new_v}{extension}"
    le_name = f"{le_base}_{new_v}{extension}"

    # Lưu tại AI folder
    joblib.dump(best_model, model_name)
    joblib.dump(le, le_name)
    print(f"\n✅ [AI] Saved: {model_name} & {le_name}")

    # Đồng bộ sang Fog Node
    fog_dir = "../smart-cushion-fog/ai/models"
    if os.path.exists(fog_dir):
        shutil.copy(model_name, os.path.join(fog_dir, model_name))
        shutil.copy(le_name, os.path.join(fog_dir, le_name))
        print(f"🚀 [FOG] Synced models to Fog Node directory.")

    return best_model, le

# ==========================================
# 3. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Bạn có thể đổi folder ở đây
    DATA_FOLDER = './lab_07052026' 
    X_all, y_all, groups_all = prepare_data(data_folder=DATA_FOLDER)
    
    if X_all is not None:
        best_xgb_model, le = train_xgb_cv(X_all, y_all, groups_all)
        
        print("\n--- TEST PREDICTION ---")
        # Sample data để kiểm tra logic
        sample_data = np.array([2846, 0, 3136, 3503, 0, 2047, 3235, 2220, 2237])
        sample_rel = sample_data / (sample_data.sum() + 1e-5)
        
        s_front = sample_rel[[0, 1, 2]].sum()
        s_back = sample_rel[[6, 7, 8]].sum()
        s_left = sample_rel[[0, 3, 6]].sum()
        s_right = sample_rel[[2, 5, 8]].sum()
        
        s_fb = (s_front - s_back) / (s_front + s_back + 1e-5)
        s_lr = (s_left - s_right) / (s_left + s_right + 1e-5)
        
        input_final = np.hstack([sample_rel, [s_fb, s_lr]]).reshape(1, -1)
        
        # Dự đoán và giải mã nhãn
        pred_encoded = int(best_xgb_model.predict(input_final)[0])
        pred_class = int(le.classes_[pred_encoded]) # Quay lại nhãn gốc (0-8)
        
        pred_prob = best_xgb_model.predict_proba(input_final)[0]
        
        print(f"Predicted Class: {POSTURE_INFO[pred_class]['name']} ({POSTURE_INFO[pred_class]['label']})")
        print(f"Confidence: {pred_prob[pred_encoded]*100:.1f}%")
