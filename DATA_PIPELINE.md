# CapyCushion AI Data Pipeline & Processing Flow

This document details the end-to-end data pipeline utilized by **CapyCushion** for raw pressure sensor data loading, noise filtering, feature engineering, and model validation.

---

## 🏗️ 1. Data Collection & Grouping

### 📂 Directory Structure
All calibration sessions are exported as Excel files (`.xlsx`) and stored in the `./data_exports` directory of the Fog Node repository. The training pipeline recursively scans this directory:
```
data_exports/
├── huong_01/
│   ├── cushion_data_NUP_20260519_111617.xlsx
│   └── cushion_data_CLL_20260519_114053.xlsx
└── vincent_01/
    ├── cushion_data_LF_20260519_120112.xlsx
    └── cushion_data_CRLL_20260519_121545.xlsx
```

### 🏷️ Target Label Detection
The target posture class is parsed dynamically from the filename using keyword matching (case-insensitive):

| Target Class | Posture ID | Filename Keywords | Posture Name |
| :--- | :---: | :--- | :--- |
| **NUP** | 0 | `straight`, `nup` | Neutral Upright Posture |
| **LF** | 1 | `leaning_forward`, `lf` | Leaning Forward |
| **LB** | 2 | `leaning_backward`, `lb` | Leaning Backward |
| **LFSR** | 3 | `support_right`, `lfsr` | Lean Forward + Support Right |
| **LFSL** | 4 | `support_left`, `lfsl` | Lean Forward + Support Left |
| **CRL** | 5 | `cross_right_ankle`, `crl` | Cross Right Leg (Ankle on Knee) |
| **CLL** | 6 | `cross_left_ankle`, `cll` | Cross Left Leg (Ankle on Knee) |
| **CRLL** | 7 | `cross_right_knee`, `crll` | Cross Right Leg (Thigh on Thigh) |
| **CLLL** | 8 | `cross_left_knee`, `clll` | Cross Left Leg (Thigh on Thigh) |

Files that do not match any known posture keywords are ignored.

### 👤 Subject Identifier Mapping
Each subdirectory directly containing the files (e.g. `vincent_01`) represents one distinct human subject. The training script maps each unique folder name to an integer Group-ID. This grouping is critical to ensure that data from the same subject is never leaked between folds during Cross-Validation.

---

## 🧹 2. Signal Quality Filtering & Noise Removal

Before any mathematical calculations, raw dataframe rows are processed through a multi-step filter designed to match the quality standards of posture research papers:

```python
def _noise_filter(df: pd.DataFrame, noise_thr: int = 20) -> pd.DataFrame:
    crop = 20
    if len(df) <= crop * 2:
        return pd.DataFrame()
    
    # Step 1: Crop transition margins
    df = df.iloc[crop:-crop].copy()
    
    # Step 2: Contact thresholding
    df = df[(df[FSR_COLS] > noise_thr).sum(axis=1) >= 3]
    if df.empty:
        return df
        
    # Step 3: Outlier amplitude filter
    tp = df[FSR_COLS].sum(axis=1)
    m  = tp.mean()
    return df[(tp >= m * 0.75) & (tp <= m * 1.25)]
```

### Step 1: Transition Margin Pruning (Crop)
- **Action**: The first `20` and the last `20` rows (frames) of each Excel sheet are pruned.
- **Rationale**: When starting or stopping a recording, the subject is in transition (e.g., sat down partially or preparing to stand up). Removing these margins eliminates unstable transient signals.

### Step 2: Contact Thresholding
- **Action**: Rows are filtered out unless **at least 3 FSR sensors** register values greater than `20` (out of 1024 ADC scale).
- **Rationale**: Eliminates frames where the seat is empty or the user is barely touching the cushion.

### Step 3: Total Pressure Stability Filter
- **Action**: Let $S_i$ be the sum of all 9 FSR sensors for row $i$, and $\mu$ be the average of $S_i$ across the entire recording session. The pipeline only retains rows satisfying:
  
  $$0.75 \times \mu \le S_i \le 1.25 \times \mu$$
- **Rationale**: Filters out sudden weight shifting anomalies, spikes, or instances where the subject temporarily lifted off the seat.

---

## 📐 3. Feature Engineering & Normalization

Once filtered, raw pressure values ($N \times 9$) are transformed to achieve weight invariance and capture physical spatial properties:

### Weight-Invariant Normalization (Sum-to-1)
To ensure the model adapts seamlessly to lightweight or heavier individuals, the raw pressure values for each row are normalized by the row's total pressure:

$$x_{\text{norm}, k} = \frac{x_k}{\sum_{j=1}^{9} x_j}$$

### Physical Feature Extraction (22-Dimensional Feature Space)
For **Random Forest** models, we compute **13 additional features** representing center of pressure, diagonal asymmetry, and region-specific weights:

1. **Center of Pressure X (CoP_X)**: Horizontal weight balance (Right sum - Left sum).
2. **Center of Pressure Y (CoP_Y)**: Vertical weight balance (Front sum - Back sum).
3. **Diagonal Difference**: Balance skewness between main diagonal (FL, MM, BR) and anti-diagonal (FR, MM, BL).
4. **Normalized standard deviation, maximum, minimum, and variance** of the FSR grid.
5. **Regional sums**: Aggregations for Front, Back, Left, Right, and Middle rows/columns.

This produces a fully normalized **22-dimensional feature vector** for training.

### 2D Grid Mapping (3×3×1)
For deep learning networks (**Tiny CNN** and **ResNet**), the 9 normalized FSR sensor values are mapped back to their original $3 \times 3$ grid structure representing the physical layout of the smart cushion:
```
┌─────────────────────────────────┐
│ Front Left │ Front Mid │ Front Right │
├────────────┼───────────┼─────────────┤
│  Mid Left  │  Mid Mid  │  Mid Right  │
├────────────┼───────────┼─────────────┤
│  Back Left │  Back Mid │  Back Right │
└─────────────────────────────────┘
```
This tensor ($3 \times 3 \times 1$) is passed directly to the Convolutional layers (`Conv2D`) to capture spatial weight distribution features.

---

## 🔬 4. Validation Strategy: LOSO CV

To prevent user-specific bias and evaluate model performance under realistic conditions, the pipeline utilizes **Leave-One-Subject-Out Cross-Validation (LOSO CV)**:

- **Method**: If there are $K$ subjects, the models are trained $K$ times. In fold $i$, subject $i$ is kept exclusively as the test set, while the remaining $K-1$ subjects are used for training.
- **Significance**: This guarantees that the test accuracy measures how well the model generalizes to a **completely new person** it has never seen before, mirroring real-world plug-and-play deployments.
- **Production Model**: Once validation metrics are evaluated, a final model is trained using the combined data of all subjects to maximize the knowledge base.
