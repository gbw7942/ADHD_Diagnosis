import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# File path
file_path = "/root/experiments/20250117_141323/metrics_epoch_50.json"

# Read JSON data
with open(file_path, "r") as f:
    data = json.load(f)

# Extract pred_classes and targets under the 'val' key
val_data = data.get("val", {})
pred_classes = val_data.get("pred_classes", [])
targets = val_data.get("targets", [])

# Ensure the data is valid
if not pred_classes or not targets:
    print("Error: pred_classes or targets data is missing in the file.")
else:
    # Compute confusion matrix
    cm = confusion_matrix(targets, pred_classes)

    # Visualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig("/root/autodl-tmp/CNNLSTM/Project/fMRI/ljhtest/confusion_matrix.png")