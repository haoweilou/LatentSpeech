import matplotlib.pyplot as plt
import pandas as pd 
dataframe = pd.read_csv("./log/loss_jukebox_upsampler2")
# dataframe = pd.read_csv("./log/loss_wavegen")

# Extract the 'epochs' column from the DataFrame

# Extract the 'epochs' column from the DataFrame
epochs = dataframe.iloc[:, 0]

# Find all columns that end with "loss"
loss_columns = [col for col in dataframe.columns if col.endswith('loss')]

# Print all loss columns
print("Loss Columns: ", loss_columns)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each loss column
for loss_col in loss_columns:
    plt.plot(epochs, dataframe[loss_col], label=loss_col)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.grid(True)
plt.show()
