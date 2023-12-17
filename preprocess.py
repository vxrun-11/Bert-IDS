import pandas as pd

# Load the KDD Cup dataset (train and test sets)
columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

train_data = pd.read_csv("D:\\NETSEC\\kddcup.data_10_percent.csv", header=None, names=columns)
test_data = pd.read_csv("D:\\NETSEC\\corrected.csv", header=None, names=columns)

# Add a column for malicious (1) or not (0) based on the label
train_data["malicious"] = train_data["label"].apply(lambda x: 1 if x != "normal." else 0)
test_data["malicious"] = test_data["label"].apply(lambda x: 1 if x != "normal." else 0)

# Select relevant columns for your text data (you can choose specific columns)
selected_columns = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
train_data = train_data[selected_columns + ["malicious"]]
test_data = test_data[selected_columns + ["malicious"]]

# Save the preprocessed data to a CSV file
train_data.to_csv("kdd_train_data.csv", index=False)
test_data.to_csv("kdd_test_data.csv", index=False)
