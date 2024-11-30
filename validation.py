# Example usage
if __name__ == "__main__":
    # List of PDF file paths to compare
    pdf_files = [
        "australia_2021.pdf",
        "singapore_2023.pdf",
        "norway_2020.pdf",
        "usa_2023.pdf",
        "iceland_2019.pdf",
    ]
    
    # Ensure all PDF files exist
    pdf_files = [pdf for pdf in pdf_files if os.path.isfile(pdf)]
    if not pdf_files:
        logging.error("No valid PDF files found for comparison.")
    else:
        similarity_results = compare_multiple_pdfs_semantically(pdf_files, threshold=95.0)
        
        # Optionally, display the similarity matrix
        import pandas as pd
        df = pd.DataFrame(similarity_results)
        print("\nSemantic Similarity Matrix (%):")
        print(df.round(2))
        
        # Save the similarity matrix to a CSV file
        df.to_csv("semantic_similarity_matrix.csv")
        logging.info("Saved similarity matrix to semantic_similarity_matrix.csv")
