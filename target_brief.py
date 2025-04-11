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
    
    # Add new columns with empty values
    new_columns = [
        "AI Research Endpoint",
        "Research Data"
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

def target_research_search(client, prompt_file_path, target_url, model="gpt-4o"):
    """
    Execute a research API call to OpenAI with web search enabled.
    
    Args:
        client: The OpenAI client
        prompt_file_path (str): Path to the file containing the research prompt
        target_url (str): The URL to search
        model (str): The OpenAI model to use for research
        
    Returns:
        The response from the OpenAI API with the researched content
    """
    print(f"\nPerforming research for target URL: {target_url} using model {model}")
    
    # Load the research prompt from file
    prompt_content = load_text_file(prompt_file_path)
    if not prompt_content:
        print("Failed to load research prompt. Cannot proceed.")
        return None
    
    try:
        # Using the responses.create method with web search
        response = client.responses.create(
            model=model,
            tools=[
                {
                    "type": "web_search_preview",
                    "search_context_size": "high"
                }
            ],
            input=f"{prompt_content}\nTarget:\n{target_url}"
        )
        
        print("Research API call completed successfully")
        return response
    except Exception as e:
        print(f"Error making research API call: {str(e)}")
        raise e

def extract_text_from_response(response):
    """
    Extract the text content from an OpenAI API response.
    
    Args:
        response: The response from the OpenAI API
        
    Returns:
        str: The extracted text from the response
    """
    response_text = ""
    
    try:
        # Try to extract text based on the response structure from the documentation
        for item in response.output:
            if hasattr(item, 'role') and item.role == 'assistant':
                for content_item in item.content:
                    if hasattr(content_item, 'text'):
                        response_text = content_item.text
                        break
        
        # If text is still empty, try an alternative approach
        if not response_text:
            for item in response.output:
                if hasattr(item, 'content'):
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            response_text += content_item.text
    except Exception as e:
        print(f"Error extracting text from response: {str(e)}")
        # Try to print the response structure for debugging
        try:
            print(f"Response structure: {response}")
            print(f"Output type: {type(response.output)}")
            print(f"Output contents: {response.output}")
        except:
            pass
    
    # Print the response for debugging
    print(f"\nResponse received (first 200 chars):\n{response_text[:200]}...\n")
    
    return response_text

def research_companies(df, research_prompt_file, client, research_model="gpt-4o"):
    """
    Process each row in the DataFrame to research the company using their URL.
    
    Args:
        df (pd.DataFrame): The DataFrame to process
        research_prompt_file (str): Path to the research prompt file
        client: The OpenAI client
        research_model (str): The model to use for research
        
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
            # Research the company
            print("\nResearching company...")
            research_response = target_research_search(client, research_prompt_file, target_dict["target_url"], model=research_model)
            
            if not research_response:
                print(f"No research data obtained for row {index+1}. Skipping.")
                continue
            
            # Extract the research text from the response
            research_text = extract_text_from_response(research_response)
            
            # Save the research data in the DataFrame
            df.at[index, "AI Research Endpoint"] = research_model
            df.at[index, "Research Data"] = research_text
            print(f"Research data preview: {research_text[:150]}...")
            
            # Save research results after each row
            df.to_csv(f"research_results_{index+1}.csv", index=False)
            print(f"Saved research results to research_results_{index+1}.csv")
            
            # Small delay to avoid rate limiting
            print(f"Waiting for 3 seconds before next company...")
            time.sleep(3)
            
        except Exception as e:
            print(f"\nError processing row {index}: {str(e)}")
            # Continue to the next row rather than failing completely
    
    print("\nAll rows processed successfully")
    return df

if __name__ == "__main__":
    # File paths
    input_file = "Test3 Growth List Startup Plan_usa_leads - Growth List Startup Plan_usa_leads.csv"
    output_file = "Growth_List_Research.csv"
    research_prompt_file = "target_brief_prompt.txt"  # File containing the research prompt
    print(f"Starting processing with input file: {input_file}")
    
    # Process the CSV file
    df = process_growth_list_csv(input_file)
    print(f"DataFrame loaded with {len(df)} rows and {len(df.columns)} columns")
    
    # Set up OpenAI API
    openai_client = setup_openai_api()
    print("OpenAI client initialized")

    # Only perform research on the companies
    updated_df = research_companies(df, research_prompt_file, openai_client, research_model="gpt-4o")
    
    # Save the updated DataFrame to a CSV file
    updated_df.to_csv(output_file, index=False)
    print(f"Updated DataFrame saved to {output_file}")