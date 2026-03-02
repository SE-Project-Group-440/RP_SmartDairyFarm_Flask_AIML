import pandas as pd
import numpy as np

file_path = "Raw-Milk-Herd-PD.xlsx"

# Load all sheets
all_sheets = pd.read_excel(file_path, sheet_name=None)

#Combine All Sheets
repro_sheets = [
    "Herd 09.07.2025",
    "Herd 01.05.2025",
    "Herd 30.01.2025",
    "Herd 24.08.2023"
]


repro_dfs = [all_sheets[sheet] for sheet in repro_sheets]
repro_df = pd.concat(repro_dfs, ignore_index=True)

print("Total cows in combined reproduction sheets:", repro_df.shape[0])


date_cols = ['Date of Birth', 'Last Caving Date', 'Caving Date', 'PD Date', 'Last Estrus/Heat Date']

# Convert each date column to datetime
def excel_serial_to_date(x):
    try:
        return pd.to_datetime('1899-12-30') + pd.to_timedelta(int(x), 'D')
    except:
        return pd.to_datetime(x, errors='coerce')

for col in date_cols:
    if col in repro_df.columns:
        repro_df[col] = repro_df[col].apply(excel_serial_to_date)

# Collect All Milk Yield Values 
milk_sheets = [
    "Milk Yield 20-21, 21-22",
    "Milk Yield 19-20, 20-21"
]

milk_values = []

for sheet in milk_sheets:
    df = all_sheets[sheet].copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    for col in df.columns:
        if col in ['S/ NO.', 'Tag No.', 'Name']:
            continue
        milk_values.extend(pd.to_numeric(df[col], errors='coerce').dropna().tolist())

milk_values = np.array(milk_values)
print(f"Total milk yield values collected: {len(milk_values)}")

np.random.seed(42)
repro_df["Milk_Yield"] = np.random.choice(milk_values, size=len(repro_df))

repro_df.to_excel("Reproduction_with_Milk.xlsx", index=False)
print("Final dataset saved to 'Reproduction_with_Milk.xlsx'")

print(repro_df.head())