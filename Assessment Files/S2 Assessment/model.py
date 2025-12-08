import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


# ============================================================
# Custom Metric (PHQ tolerance accuracy)
# ============================================================
def phq_tolerance_acc(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.abs(y_true - y_pred) <= 3.0, tf.float32))


# ============================================================
# Configuration
# ============================================================
DATA_PATH = "ph9Dataset.xlsx"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TOKENS = 20000
MAX_LEN = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 64
EPOCHS = 20
BATCH_SIZE = 32


# ============================================================
# Helpers
# ============================================================
def sanitize_name(name: str) -> str:
    s = name.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    if len(s) == 0:
        s = "col"
    if re.match(r"^[0-9]", s):
        s = "f_" + s
    return s


def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ============================================================
# 1. Load dataset
# ============================================================
df = pd.read_excel(DATA_PATH)
df = df.dropna(axis=1, how="all")

text_cols = [c for c in df.columns if df[c].dtype == object][:9]
if len(text_cols) < 9:
    raise ValueError(f"Need 9 text fields, found {len(text_cols)}")

print("Detected text fields:")
for c in text_cols:
    print(" •", c)

sanitized = {orig: sanitize_name(orig) for orig in text_cols}

# Detect PHQ column
phq_col = None
for c in df.select_dtypes(include=[np.number]).columns:
    if 0 <= df[c].min() <= 30 and df[c].max() <= 30:
        phq_col = c
        break
if phq_col is None:
    raise ValueError("PHQ column not detected.")
print("\nDetected PHQ column:", phq_col)

# Detect severity
severity_col = None
for c in df.columns:
    if c in text_cols or c == phq_col:
        continue
    if df[c].dtype == object and df[c].nunique() <= 12:
        severity_col = c
        break
    if np.issubdtype(df[c].dtype, np.integer) and df[c].nunique() <= 12:
        severity_col = c
        break
if severity_col is None:
    raise ValueError("Severity column not detected.")
print("Detected Severity column:", severity_col)


# ============================================================
# 2. Clean text fields
# ============================================================
for col in text_cols:
    df[col] = df[col].fillna("").apply(clean_text)


# ============================================================
# 3. Prepare targets
# ============================================================
y_phq = df[phq_col].astype(float).values
severity_raw = df[severity_col].astype(str).values

severity_encoder = LabelEncoder()


# ============================================================
# 4. Train/Test split
# ============================================================
n = len(df)
indices = np.arange(n)

unique_sev = np.unique(severity_raw)
stratify = severity_raw if len(unique_sev) > 1 else None

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=stratify
)

print("\nTrain samples:", len(train_idx))
print("Test samples:", len(test_idx))


# ============================================================
# 5. Text Vectorization
# ============================================================
text_vectorizers = {}
X_train = {}
X_test = {}

for orig_col in text_cols:
    safe = sanitized[orig_col]

    vec = layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=MAX_LEN,
        name=f"vect_{safe}"
    )

    train_texts = df[orig_col].values[train_idx]
    vec.adapt(train_texts)
    text_vectorizers[orig_col] = vec

    X_train[f"input_{safe}"] = vec(train_texts).numpy().astype("int32")
    X_test[f"input_{safe}"] = vec(df[orig_col].values[test_idx]).numpy().astype("int32")

# Encode severity labels
y_sev_train_raw = severity_raw[train_idx]
y_sev_test_raw  = severity_raw[test_idx]

severity_encoder.fit(y_sev_train_raw)
y_sev_train = severity_encoder.transform(y_sev_train_raw).astype("int32")
y_sev_test  = severity_encoder.transform(y_sev_test_raw).astype("int32")
y_phq_train = y_phq[train_idx].astype("float32")
y_phq_test  = y_phq[test_idx].astype("float32")

num_classes = len(severity_encoder.classes_)
print("\nSeverity categories:", list(severity_encoder.classes_))


# ============================================================
# 6. Build model
# ============================================================
inputs = []
encoded = []

for orig_col in text_cols:
    safe = sanitized[orig_col]
    inp = layers.Input(shape=(MAX_LEN,), dtype="int32", name=f"input_{safe}")
    emb = layers.Embedding(MAX_TOKENS, EMBEDDING_DIM, name=f"embed_{safe}")(inp)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=False),
        name=f"bilstm_{safe}"
    )(emb)
    x = layers.Dropout(0.3, name=f"dropout_{safe}")(x)

    inputs.append(inp)
    encoded.append(x)

merged = layers.Concatenate(name="concat_all")(encoded)
h = layers.Dense(128, activation="relu")(merged)
h = layers.Dropout(0.3)(h)

phq_out = layers.Dense(1, name="phq_output")(h)
sev_out = layers.Dense(num_classes, activation="softmax", name="severity_output")(h)

model = Model(inputs=inputs, outputs=[phq_out, sev_out])


# ============================================================
# 7. Compile
# ============================================================
model.compile(
    optimizer="adam",
    loss={
        "phq_output": "mse",
        "severity_output": "sparse_categorical_crossentropy"
    },
    metrics={
        "phq_output": ["mse", phq_tolerance_acc],
        "severity_output": ["accuracy"]
    }
)

input_names = [inp.name.split(":")[0] for inp in model.inputs]
print("\nModel Inputs:", input_names)


# ============================================================
# 8. Training dictionaries
# ============================================================
train_inputs = {name: X_train[name] for name in input_names}
test_inputs  = {name: X_test[name]  for name in input_names}


# ============================================================
# 9. Train
# ============================================================
history = model.fit(
    train_inputs,
    {
        "phq_output": y_phq_train,
        "severity_output": y_sev_train
    },
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)


# ============================================================
# 10. Evaluate
# ============================================================
print("\nEvaluating model on test set...")
model.evaluate(
    test_inputs,
    {
        "phq_output": y_phq_test,
        "severity_output": y_sev_test
    },
    verbose=2
)



# ============================================================
# 11. SIMPLE READABLE FINAL OUTPUT (Replaces Confusion Matrices)
# ============================================================
print("\nComputing final prediction statistics...")

# ---- Get Predictions ----
pred_phq, pred_sev = model.predict(test_inputs, verbose=0)

pred_phq = pred_phq.flatten()
pred_sev_classes = np.argmax(pred_sev, axis=1)

# ============================================================
# 1. Severity Prediction Accuracy
# ============================================================
total_sev = len(y_sev_test)
correct_sev = np.sum(pred_sev_classes == y_sev_test)
incorrect_sev = total_sev - correct_sev

print("\n===============================")
print(" SEVERITY PREDICTION SUMMARY")
print("===============================\n")
print(f"Total predictions:     {total_sev}")
print(f"Correct predictions:   {correct_sev}")
print(f"Incorrect predictions: {incorrect_sev}")
print(f"Accuracy:              {correct_sev / total_sev * 100:.3f}%")


# ============================================================
# 2. PHQ-9 Prediction Accuracy (using ±3 tolerance)
# ============================================================
phq_diffs = np.abs(pred_phq - y_phq_test)
correct_phq = np.sum(phq_diffs <= 3)
total_phq = len(y_phq_test)
incorrect_phq = total_phq - correct_phq

print("\n===========================================")
print(" PHQ-9 PREDICTION SUMMARY (±3 tolerance)")
print("===========================================\n")
print(f"Total predictions:     {total_phq}")
print(f"Correct predictions:   {correct_phq}   (within ±3)")
print(f"Incorrect predictions: {incorrect_phq} (outside ±3)")
print(f"Tolerance Accuracy:    {correct_phq / total_phq * 100:.3f}%")