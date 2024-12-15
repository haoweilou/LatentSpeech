import matplotlib.pyplot as plt
import pandas as pd 
dataframe = pd.read_csv("./log/loss_StyleSpeech2_FF")
dataframe = pd.read_csv("./log/loss_aligner")
dataframe = pd.read_csv("./log/loss_StyleSpeechDiff")

print(dataframe)
# dataframe = pd.read_csv("./log/loss_wavegen")

# Extract the 'epochs' column from the DataFrame

# Extract the 'epochs' column from the DataFrame
epochs = dataframe.iloc[:, 0]
print(epochs)

# Find all columns that end with "loss"
loss_columns = [col for col in dataframe.columns if col.endswith('loss')]

# Print all loss columns
print("Loss Columns: ", loss_columns)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each loss column
for loss_col in loss_columns:
    plt.plot(epochs, dataframe[loss_col], label=loss_col,scaley=True)

# plt.ylim(0, .1)


plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()
plt.grid(True)
plt.show()
