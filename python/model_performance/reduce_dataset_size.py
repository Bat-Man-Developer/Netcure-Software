import pandas as pd

# Function to read multiple CSV files, delete 95% of the rows from each, and save the results
def reduce_data_in_multiple_csv(input_files, output_files):
    for input_file, output_file in zip(input_files, output_files):
        # Read the CSV file
        df = pd.read_csv(input_file, low_memory=False)

        # Calculate 95% of the total number of rows
        num_rows_to_delete = int(len(df) * 0.95)

        # Delete the specified number of rows from the beginning
        df_reduced = df.iloc[num_rows_to_delete:]

        # Save the modified DataFrame to a new CSV file
        df_reduced.to_csv(output_file, index=False)
        print(f"Deleted {num_rows_to_delete} rows (95% of the data) from {input_file} and saved to {output_file}")

# Example usage
input_files = [
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_DNS.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_LDAP.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_MSSQL.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_NetBIOS.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_NTP.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_SNMP.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_SSDP.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/DrDoS_UDP.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/Syn.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/TFTP.csv',
    'CIC 2019 DDOS DATASET/reduced_50000_rows/UDPLag.csv'
]

output_files = [
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_DNS.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_LDAP.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_MSSQL.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_NetBIOS.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_NTP.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_SNMP.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_SSDP.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/DrDoS_UDP.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/Syn.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/TFTP.csv',
    'CIC 2019 DDOS DATASET/reduced_10000_rows/UDPLag.csv'
]

reduce_data_in_multiple_csv(input_files, output_files)