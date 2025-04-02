import os
import pandas as pd
import tiktoken
from dotenv import load_dotenv

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Counts the number of tokens in the given text using the tiktoken library.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception as e:
        print(f"Could not get encoding for model {model}: {e}. Falling back to 'cl100k_base'.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    return len(tokens)

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file from the given file path and returns it as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def save_df_with_suffix(df: pd.DataFrame, original_file_path: str, suffix: str) -> str:
    """
    Saves the given DataFrame to a new CSV file by appending the specified suffix before the file extension.
    Returns the new file path.
    """
    base, ext = os.path.splitext(original_file_path)
    new_file_path = f"{base}{suffix}{ext}"
    df.to_csv(new_file_path, index=False)
    print(f"Data saved to: {new_file_path}")
    return new_file_path

def process_csv_outputs(file_path: str):
    """
    Processes the CSV file to create two outputs:
      1. USA Leads: Rows where "Country" is exactly "United States".
      2. Everyone Minus India & China: Rows that are not "United States", "India", or "China".
    Saves both outputs as new CSV files.
    """
    df = load_csv(file_path)
    
    # Filter for USA leads: exact match on "United States"
    df_usa = df[df["Country"] == "United States"]
    
    # Filter for everyone else minus India and China.
    # This includes only rows where "Country" is not United States, India, or China.
    df_everyone = df[(df["Country"] != "United States") & (df["Country"] != "India") & (df["Country"] != "China")]
    
    # Save the outputs
    usa_file = save_df_with_suffix(df_usa, file_path, "_usa_leads")
    everyone_file = save_df_with_suffix(df_everyone, file_path, "_everyone_minus_Ind&china")
    
    return usa_file, everyone_file

if __name__ == "__main__":
    # Load environment variables if needed (e.g., for API keys)
    load_dotenv()
    
    # Manually specify the CSV file path
    file_path = "Growth List Startup Plan.csv"
    
    # Process the CSV to create the two outputs
    usa_csv, everyone_csv = process_csv_outputs(file_path)
    
    # Optional: Count tokens for each file's content (if you plan to use them in prompts)
    with open(usa_csv, "r") as f:
        usa_content = f.read()
    with open(everyone_csv, "r") as f:
        everyone_content = f.read()
    
    print("Token count for USA CSV:", count_tokens(usa_content))
    print("Token count for Everyone Minus India & China CSV:", count_tokens(everyone_content))

