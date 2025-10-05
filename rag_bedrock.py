import boto3
import json
from typing import Dict, Any

# --- Configuration ---
# IMPORTANT: Replace the placeholder below with your actual Bedrock Knowledge Base ID.
KNOWLEDGE_BASE_ID = "######"  # e.g., 'A1B2C3D4E5'

# IMPORTANT: SET THE AWS REGION WHERE YOUR KNOWLEDGE BASE IS DEPLOYED.
# This must match the region of your KB (e.g., 'us-east-1', 'us-west-2').
REGION_NAME = "us-west-2"

# Model used for the generation part of RAG. Using Claude 3 Haiku which is widely available.
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Alternative model IDs you can try if the above doesn't work:
# MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
# MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
# MODEL_ID = "meta.llama3-8b-instruct-v1:0"  # If you have access


def check_model_access() -> bool:
    """
    Check if the configured model is accessible in the specified region.
    
    Returns:
        bool: True if model is accessible, False otherwise.
    """
    try:
        client = boto3.client('bedrock', region_name=REGION_NAME)
        response = client.list_foundation_models()
        
        available_models = [model['modelId'] for model in response['modelSummaries']]
        
        if MODEL_ID in available_models:
            print(f"✅ Model {MODEL_ID} is available in {REGION_NAME}")
            return True
        else:
            print(f"❌ Model {MODEL_ID} is not available in {REGION_NAME}")
            print(f"Available models: {available_models[:5]}...")  # Show first 5
            return False
            
    except Exception as e:
        print(f"❌ Error checking model access: {e}")
        return False


def run_rag_query(query: str) -> Dict[str, Any]:
    """
    Executes a RAG (Retrieval-Augmented Generation) query using the
    Amazon Bedrock Agent Runtime API's retrieveAndGenerate function.

    Args:
        query (str): The user's prompt or question.

    Returns:
        Dict[str, Any]: A dictionary containing the generated text, sources, or an error message.
    """
    if KNOWLEDGE_BASE_ID == "YOUR_KNOWLEDGE_BASE_ID_HERE":
        return {"error": "Configuration Missing: Please set KNOWLEDGE_BASE_ID."}
    if REGION_NAME == "YOUR_KB_REGION_HERE":
        return {"error": "Configuration Missing: Please set REGION_NAME to your Knowledge Base region (e.g., 'us-east-1')."}

    print(f"\n--- Sending query to Bedrock RAG: '{query}' ---")
    print(f"Using model: {MODEL_ID}")
    print(f"Region: {REGION_NAME}")
    print(f"Knowledge Base ID: {KNOWLEDGE_BASE_ID}")

    try:
        # Initialize the client with the specified region
        client = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=REGION_NAME
        )

        # The retrieveAndGenerate call encapsulates the full RAG process
        response = client.retrieve_and_generate(
            input={'text': query},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    # Dynamically construct the model ARN using the configured region
                    'modelArn': f'arn:aws:bedrock:{REGION_NAME}::foundation-model/{MODEL_ID}',
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 3  # Number of source documents to retrieve
                        }
                    }
                }
            }
        )

        # --- Process the response ---

        # 1. Extract the Generated Answer
        generated_text = response['output']['text']

        # 2. Extract Source/Citation Information
        citations = []
        if 'citations' in response:
            for citation in response['citations']:
                if 'retrievedReferences' in citation:
                    for ref in citation['retrievedReferences']:
                        # The URI points to the S3 object/source file
                        uri = ref.get('uri', 'N/A')
                        # The text is the snippet of the document used as context
                        content = ref['content']['text']
                        citations.append({
                            "source_uri": uri,
                            "snippet": content
                        })

        return {
            "answer": generated_text,
            "sources": citations,
            "model_used": MODEL_ID.split('/')[-1]
        }

    except Exception as e:
        error_msg = str(e)
        print(f"An error occurred during the Bedrock API call: {e}")
        
        # Provide specific guidance for common errors
        if "ValidationException" in error_msg and "access to the model" in error_msg:
            return {
                "error": f"Model Access Error: {error_msg}\n\nTroubleshooting:\n1. Check if the model '{MODEL_ID}' is available in region '{REGION_NAME}'\n2. Verify you have access to this model in your AWS account\n3. Try using a different model like 'anthropic.claude-3-haiku-20240307-v1:0' or 'anthropic.claude-3-sonnet-20240229-v1:0'"
            }
        elif "AccessDeniedException" in error_msg:
            return {
                "error": f"Access Denied: {error_msg}\n\nTroubleshooting:\n1. Check your AWS credentials\n2. Verify you have the necessary IAM permissions for Bedrock\n3. Ensure your Knowledge Base ID is correct"
            }
        else:
            return {"error": error_msg}

# --- Example Usage ---

if __name__ == "__main__":
    print("--- Amazon Bedrock RAG Chatbot Demo ---")
    print("Prerequisites: AWS credentials configured and a valid Bedrock Knowledge Base ID and REGION_NAME set.")
    print("-" * 40)

    # Check model access first
    print("Checking model access...")
    if not check_model_access():
        print("\n⚠️  Model access check failed. The script will still attempt to run, but may fail.")
        print("You can try changing the MODEL_ID to one of the alternatives in the code.")
        print("-" * 40)

    # Example query
    test_query = "Which things did author learnt and understood with his experience?"

    rag_result = run_rag_query(test_query)

    if 'error' in rag_result:
        print(f"\n❌ Failed to run RAG query. Error: {rag_result['error']}")
    else:
        print("\n✅ RAG Query Successful!")
        print("-" * 40)
        print(f"🤖 **Generated Answer** (via {rag_result['model_used']}):")
        print(rag_result['answer'])
        print("-" * 40)

        if rag_result['sources']:
            print(f"📚 **Sources Used** ({len(rag_result['sources'])} snippets):")
            for i, source in enumerate(rag_result['sources']):
                print(f"  [{i+1}] Source URI: {source['source_uri']}")
                print(f"      Snippet: {source['snippet'][:150].strip()}...")
            print("-" * 40)
        else:
            print("No specific sources were cited or retrieved.")
