import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch  # Import torch to handle PyTorch tensors
import numpy as np
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import string

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the SBERT model (consider upgrading to a more powerful model if needed)
model = SentenceTransformer('all-mpnet-base-v2')  # Example of a more powerful model

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file: str) -> str:
    logging.info(f"Extracting text from {pdf_file}")
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logging.error(f"Error reading {pdf_file}: {e}")
    return text

# Function to split text into sentences for finer granularity
def split_into_sentences(text: str) -> List[str]:
    logging.debug("Splitting text into sentences")
    sentences = sent_tokenize(text)
    return sentences

# Function to clean the sentences
def clean_sentences(sentences: List[str]) -> List[str]:
    logging.debug("Cleaning sentences")
    cleaned = []
    translator = str.maketrans('', '', string.punctuation)
    for sentence in sentences:
        # Remove non-printable characters
        sentence = ''.join(filter(lambda x: x.isprintable(), sentence))
        # Remove punctuation
        sentence = sentence.translate(translator)
        # Convert to lowercase
        sentence = sentence.lower()
        # Remove excess whitespace
        sentence = ' '.join(sentence.split())
        # Filter out short sentences
        if len(sentence) > 5:
            cleaned.append(sentence)
    return cleaned

# Function to process a single PDF and return its cleaned sentences
def process_pdf(pdf_file: str) -> List[str]:
    text = extract_text_from_pdf(pdf_file)
    sentences = split_into_sentences(text)
    cleaned_sentences = clean_sentences(sentences)
    return cleaned_sentences

# Function to encode sentences and cache embeddings
def encode_sentences(sentences: List[str]) -> torch.Tensor:
    if not sentences:
        return torch.tensor([])
    logging.info(f"Encoding {len(sentences)} sentences")
    embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
    return embeddings

# Function to generate a single document embedding by averaging sentence embeddings
def generate_document_embedding(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.numel() == 0:
        return torch.tensor([])
    document_embedding = torch.mean(embeddings, dim=0)
    return document_embedding

# Function to calculate cosine similarity between two document embeddings
def calculate_cosine_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    if embedding1.numel() == 0 or embedding2.numel() == 0:
        return 0.0
    similarity = util.cos_sim(embedding1, embedding2).item()
    return similarity * 100  # Convert to percentage

# Main function to process multiple PDFs and compute similarity matrix
def compare_multiple_pdfs_semantically(pdf_paths: List[str], threshold: float = 95.0) -> Dict[str, Dict[str, float]]:
    logging.info("Processing multiple PDFs for semantic similarity")
    
    # Step 1: Extract and process all PDFs
    sentences_dict = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_paths}
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                sentences = future.result()
                sentences_dict[pdf] = sentences
                logging.info(f"Processed {pdf} with {len(sentences)} sentences")
            except Exception as e:
                logging.error(f"Error processing {pdf}: {e}")
    
    # Step 2: Encode all sentences and generate document embeddings
    embeddings_dict = {}
    for pdf, sentences in sentences_dict.items():
        sentence_embeddings = encode_sentences(sentences)
        document_embedding = generate_document_embedding(sentence_embeddings)
        embeddings_dict[pdf] = document_embedding
    
    # Step 3: Compute pairwise similarities
    similarity_matrix = {}
    pdf_list = list(pdf_paths)
    for i, pdf1 in enumerate(pdf_list):
        similarity_matrix[pdf1] = {}
        for j, pdf2 in enumerate(pdf_list):
            if i == j:
                similarity_matrix[pdf1][pdf2] = 100.0
            elif j < i:
                # Symmetric matrix
                similarity_matrix[pdf1][pdf2] = similarity_matrix[pdf2][pdf1]
            else:
                embedding1 = embeddings_dict[pdf1]
                embedding2 = embeddings_dict[pdf2]
                if embedding1.numel() == 0 or embedding2.numel() == 0:
                    similarity = 0.0
                    logging.warning(f"No valid sentences to compare between {pdf1} and {pdf2}")
                else:
                    similarity = calculate_cosine_similarity(embedding1, embedding2)
                similarity_matrix[pdf1][pdf2] = similarity
                logging.info(f"Similarity between {pdf1} and {pdf2}: {similarity:.2f}%")
    
    # Step 4: Analyze similarities based on the threshold
    for pdf1 in pdf_paths:
        for pdf2 in pdf_paths:
            if pdf1 != pdf2:
                sim = similarity_matrix[pdf1][pdf2]
                if sim < threshold:
                    logging.info(f"Significant difference between {pdf1} and {pdf2}: {100 - sim:.2f}% different")
                else:
                    logging.info(f"{pdf1} and {pdf2} are conceptually similar.")
    
    return similarity_matrix

# Example usage
if __name__ == "__main__":
    # List of PDF file paths to compare
    pdf_files = [
        "singapore_2023.pdf",
        "norway_2020.pdf",
        "usa_2023.pdf",
        "iceland_2019.pdf",
        # Add more PDFs as needed
    ]
    
    # Ensure all PDF files exist
    pdf_files = [pdf for pdf in pdf_files if os.path.isfile(pdf)]
    if not pdf_files:
        logging.error("No valid PDF files found for comparison.")
    else:
        similarity_results = compare_multiple_pdfs_semantically(pdf_files, threshold=95.0)
        
        # Display the similarity matrix
        df = pd.DataFrame(similarity_results, index=pdf_files, columns=pdf_files)
        print("\nSemantic Similarity Matrix (%):")
        print(df.round(2))
        
        # Save the similarity matrix to a CSV file
        df.to_csv("semantic_similarity_matrix.csv")
        logging.info("Saved similarity matrix to semantic_similarity_matrix.csv")
