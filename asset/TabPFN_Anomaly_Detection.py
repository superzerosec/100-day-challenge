import torch
from sklearn.datasets import load_breast_cancer
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

# Set device - try GPU when available, fall back to CPU if there are device compatibility issues
# Note: The unsupervised extension may have device compatibility issues where model components
# end up on different devices. Data tensors are kept on CPU because the library converts them to numpy internally.
data_device = "cpu"  # Keep data on CPU for numpy conversion in library

# Try GPU first if available
model_device = "cuda" if torch.cuda.is_available() else "cpu"
if model_device == "cuda":
    print("Attempting to use GPU for models...")
else:
    print("Using CPU (GPU not available)")

# Load dataset
df = load_breast_cancer(return_X_y=False)
X = df["data"]
attribute_names = df["feature_names"]

# Initialize models
clf = TabPFNClassifier(device=model_device, n_estimators=4)
reg = TabPFNRegressor(device=model_device, n_estimators=4)
model_unsupervised = TabPFNUnsupervisedModel(
    tabpfn_clf=clf, tabpfn_reg=reg
)

# Compute anomaly scores
model_unsupervised.fit(X)

# Convert X to tensor for outliers() method
# Keep on CPU because library converts to numpy internally
X_tensor = torch.tensor(X, dtype=torch.float32).to(data_device)

# Convert the stored X_ to tensor to avoid errors in outliers()
# The outliers() method uses self.X_ internally and needs it to be a tensor
# Keep on CPU because library converts to numpy internally
if hasattr(model_unsupervised, 'X_'):
    if not isinstance(model_unsupervised.X_, torch.Tensor):
        model_unsupervised.X_ = torch.tensor(model_unsupervised.X_, dtype=torch.float32).to(data_device)
    else:
        # Ensure it's on CPU and contiguous (library needs CPU for numpy conversion)
        model_unsupervised.X_ = model_unsupervised.X_.to(data_device).contiguous()

# Try to compute outlier scores, fall back to CPU if device mismatch occurs
try:
    scores = model_unsupervised.outliers(
        X_tensor,
        n_permutations=10,
    )
    print("Outlier scores:", scores[:10])
except RuntimeError as e:
    if "device" in str(e).lower() or "cuda" in str(e).lower():
        print(f"Device mismatch error with GPU: {e}")
        print("Falling back to CPU...")
        # Reinitialize models on CPU
        clf = TabPFNClassifier(device="cpu", n_estimators=4)
        reg = TabPFNRegressor(device="cpu", n_estimators=4)
        model_unsupervised = TabPFNUnsupervisedModel(
            tabpfn_clf=clf, tabpfn_reg=reg
        )
        model_unsupervised.fit(X)
        # Ensure X_ is on CPU
        if hasattr(model_unsupervised, 'X_'):
            if not isinstance(model_unsupervised.X_, torch.Tensor):
                model_unsupervised.X_ = torch.tensor(model_unsupervised.X_, dtype=torch.float32).to(data_device)
            else:
                model_unsupervised.X_ = model_unsupervised.X_.to(data_device).contiguous()
        # Retry with CPU
        scores = model_unsupervised.outliers(
            X_tensor,
            n_permutations=10,
        )
        print("Outlier scores (CPU):", scores[:10])
    else:
        raise  # Re-raise if it's a different error