import pandas as pd
import os
import time
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import re

def process_growth_list_csv(input_file_path):
    """
    Load a CSV file into a pandas DataFrame and add specified columns.
    
    Args:
        input_file_path (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: The modified DataFrame
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file_path)
    
    # Add the six new columns with empty values
    new_columns = [
        "AI Copy Generation Endpoint", 
        "Subject 1", 
        "Subject 2", 
        "Subject 3",
        "Subject 4", 
        "Body"
    ]
    
    for column in new_columns:
        df[column] = ""
    
    return df       

def load_text_file(file_path: str) -> str:
    """
    Load the content of a text file and return it as a string.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: The content of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Successfully loaded text from {file_path}")
        return content

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return ""

def setup_openai_api():
    """
    Set up the OpenAI API with the API key from environment variables.
    Returns an initialized OpenAI client.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please check your .env file.")
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)
    print("OpenAI API configured successfully")
    return client

def execute_api_call(client, prompt_content, target_url):
    """
    Execute an API call to OpenAI with web search enabled using high context.
    
    Args:
        client: The OpenAI client
        prompt_content (str): The content to use as a prompt
        target_url (str): The URL to search
        
    Returns:
        The response from the OpenAI API
    """
    print(f"\nMaking OpenAI API call for target URL: {target_url}")
    
    try:
        # Using the responses.create method with web search as shown in the documentation
        # and setting search_context_size to "high" for better search results
        response = client.responses.create(
            model="gpt-4o",
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "high"  # Use high context for better results
                }
            ],
            input=f"{prompt_content}\nTarget:\n{target_url}"
        )
        
        print("OpenAI API call completed successfully")
        return response
    except Exception as e:
        print(f"Error making OpenAI API call: {str(e)}")
        raise e

def parse_response(response):
    """
    Parse the response from the OpenAI API to extract subjects and body.
    
    Args:
        response: The response from the OpenAI API
        
    Returns:
        dict: A dictionary containing the subjects and body
    """
    # Get the response text directly from the output_text property
    response_text = response.output_text
    
    # Print the response for debugging
    print(f"\nResponse received (first 200 chars):\n{response_text[:200]}...\n")
    
    # Extract the subjects and body using regex
    subjects = []
    subject_pattern = r"Subject (?:Option )?(\d+): (.*?)(?:\n|$)"
    body_pattern = r"Hey \[Target\],([\s\S]*?)(?:\nCall me anytime|$)"
    
    # Extract subjects
    subject_matches = re.findall(subject_pattern, response_text)
    for _, subject_text in subject_matches:
        subjects.append(subject_text.strip())
    
    # Ensure we have exactly 4 subjects (pad with empty strings if needed)
    while len(subjects) < 4:
        subjects.append("")
    
    # Extract body
    body_match = re.search(body_pattern, response_text)
    body = body_match.group(1).strip() if body_match else ""
    
    print(f"Extracted {len(subjects)} subjects and body text of length {len(body)}")
    
    return {
        "subjects": subjects[:4],
        "body": body
    }

def openai_call(df: pd.DataFrame, prompt, client):
    """
    Process each row in the DataFrame, call OpenAI API, and update the DataFrame with results.
    
    Args:
        df (pd.DataFrame): The DataFrame to process
        prompt (str): The prompt template to use
        client: The OpenAI client
        
    Returns:
        pd.DataFrame: The updated DataFrame
    """
    total_rows = len(df)
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Print progress
        print(f"\n--- Processing row {index+1}/{total_rows} ---")
        
        # Skip if URL is missing
        if pd.isna(row["URL"]) or row["URL"] == "":
            print(f"Skipping row {index+1} due to missing URL")
            continue
        
        # Create target dictionary
        target_dict = {
            "target_url": row["URL"] if pd.notna(row["URL"]) else "",
            "ceo_name": row["CEO Name"] if pd.notna(row["CEO Name"]) else "",
            "ceo_email": row["CEO Email"] if pd.notna(row["CEO Email"]) else ""
        }
        
        print(f"Target URL: {target_dict['target_url']}")
        print(f"CEO Name: {target_dict['ceo_name']}")
        
        try:
            # Call OpenAI API with prompt, target URL
            response = execute_api_call(client, prompt, target_dict["target_url"])
            
            # Parse the response to extract subjects and body
            parsed_data = parse_response(response)
            
            # Get the CEO's first name (first word in CEO Name)
            ceo_first_name = target_dict["ceo_name"].split()[0] if target_dict["ceo_name"] else "[Target]"
            print(f"Using CEO first name: {ceo_first_name}")
            
            # Replace [Target] with CEO's first name in the body
            body_text = parsed_data["body"].replace("[Target]", ceo_first_name)
            
            # Update the DataFrame with the results
            df.at[index, "AI Copy Generation Endpoint"] = "gpt-4o"
            
            # Add subjects to the DataFrame
            for i, subject in enumerate(parsed_data["subjects"]):
                df.at[index, f"Subject {i+1}"] = subject
                print(f"Subject {i+1}: {subject}")
            
            # Add body to the DataFrame
            df.at[index, "Body"] = body_text
            print(f"Body preview: {body_text[:100]}...")
            
            # Save intermediate results after each successful row
            df.to_csv(f"interim_results_{index+1}.csv", index=False)
            print(f"Saved interim results to interim_results_{index+1}.csv")
            
            # Small delay to avoid rate limiting
            print(f"Waiting for 3 seconds before next API call...")
            time.sleep(3)
            
        except Exception as e:
            print(f"\nError processing row {index}: {str(e)}")
            # Continue to the next row rather than failing completely
    
    print("\nAll rows processed successfully")
    return df

if __name__ == "__main__":
    # File paths
    input_file = "Test3 Growth List Startup Plan_usa_leads - Growth List Startup Plan_usa_leads.csv"
    output_file = "Growth_List_copy.csv"
    prompt_file = "CombinedPrompt.txt"  # File containing the prompt template
    
    print(f"Starting processing with input file: {input_file}")
    
    # Process the CSV file
    df = process_growth_list_csv(input_file)
    print(f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns")
    
    # Load the prompt template
    prompt = load_text_file(prompt_file)
    
    # Set up OpenAI API
    openai_client = setup_openai_api()
    print("OpenAI client initialized")

    # Process the DataFrame with OpenAI API calls
    updated_df = openai_call(df, prompt, openai_client)
    
    # Save the updated DataFrame to a CSV file
    updated_df.to_csv(output_file, index=False)
    print(f"Updated DataFrame saved to {output_file}")

