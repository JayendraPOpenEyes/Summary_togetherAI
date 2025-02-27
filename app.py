import streamlit as st
import pandas as pd
from summarize_togetherAI import process_input
import subprocess
from io import BytesIO
import logging
import re
from PyPDF2 import PdfReader
import time

# Configure logging at INFO level to see detailed output
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_poppler_installed():
    """Check if Poppler is installed and accessible."""
    try:
        result = subprocess.run(
            ["pdftoppm", "-v"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if "pdftoppm" in result.stderr or "pdftoppm" in result.stdout:
            print("✅ Poppler is installed and accessible.")
            return True
        else:
            print("❌ Poppler is NOT installed or not in PATH.")
            return False
    except FileNotFoundError:
        print("❌ Poppler is NOT installed or not in PATH.")
        return False

check_poppler_installed()

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyPDF2 (for fallback purposes)."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def typewriter_effect(text, placeholder, delay=0.005):
    """Simulate a typewriter effect by displaying text character by character."""
    display_text = ""
    for char in text:
        display_text += char
        placeholder.markdown(display_text)
        time.sleep(delay)

def display_summary(summary, url, use_typewriter=False):
    """Display the summary in a structured format with optional typewriter effect."""
    with st.expander(f"Summary for {url}", expanded=True):
        st.subheader("Summary")
        if "Error" in summary["summary"]:
            st.warning(f"Could not generate summary: {summary['summary']}")
        elif use_typewriter:
            placeholder = st.empty()
            typewriter_effect(summary["summary"], placeholder)
        else:
            st.markdown(summary["summary"])
        st.write("---")

def main():
    st.title("Stateside Bill Summarization")
    st.write("Enter a URL of a Stateside bill to summarize its content, upload an Excel file, or upload a PDF file.")

    # Add a text area for custom prompt
    custom_prompt = st.text_area("Enter your custom prompt (optional):", height=150, placeholder="e.g., 'Summarize this in 3 sentences'")

    # Initialize session state for persistent data
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'prev_upload' not in st.session_state:
        st.session_state.prev_upload = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'all_summaries' not in st.session_state:
        st.session_state.all_summaries = {}
    if 'last_processed_url' not in st.session_state:
        st.session_state.last_processed_url = ""

    # Individual URL processing
    url = st.text_input("Enter URL")
    if st.button("Summarize URL"):
        if url:
            with st.spinner('Processing...'):
                result = process_input(url, custom_prompt=custom_prompt)
                logging.info(f"Result from process_input: {result}")
                if isinstance(result, dict) and "error" in result:
                    st.warning(f"Failed to process URL: {result['error']}")
                elif isinstance(result, dict) and "summary" in result:
                    st.success("Summarization complete!")
                    st.session_state.all_summaries[url] = {"summary": result["summary"]}
                    st.session_state.last_processed_url = url
                    display_summary(st.session_state.all_summaries[url], url, use_typewriter=True)
                else:
                    st.error("An unexpected error occurred: Invalid result format.")
        else:
            st.error("Please enter a valid URL.")

    # Display URL summaries (without typewriter effect for previously processed URLs)
    if st.session_state.all_summaries:
        st.subheader("URL Summaries")
        for url, summary in st.session_state.all_summaries.items():
            if url != st.session_state.last_processed_url:
                display_summary(summary, url)

    # Excel file processing
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if uploaded_file is not None:
        if st.session_state.prev_upload != uploaded_file.name:
            st.session_state.processed_df = pd.read_excel(uploaded_file)
            st.session_state.prev_upload = uploaded_file.name
            st.session_state.processing_complete = False
            st.session_state.all_summaries = {}
            if 'Summary' not in st.session_state.processed_df.columns:
                st.session_state.processed_df['Summary'] = ''
        df = st.session_state.processed_df
        if 'BillState' not in df.columns or 'BillTextURL' not in df.columns:
            st.error("File must contain 'BillState' and 'BillTextURL' columns")
        else:
            if st.button("Process Excel File"):
                with st.spinner('Processing URLs... This may take several minutes'):
                    total_urls = len(df['BillTextURL'].dropna())
                    processed_count = 0
                    for index, row in df.iterrows():
                        url = row['BillTextURL']
                        if pd.notna(url) and df.at[index, 'Summary'] == '':
                            processed_count += 1
                            try:
                                result = process_input(url, custom_prompt=custom_prompt)
                                status_msg = f"Processing URL {processed_count}/{total_urls}: {url}"
                                if isinstance(result, dict) and "error" in result:
                                    st.warning(f"{status_msg} - Can't generate summary: {result['error']}")
                                    df.at[index, 'Summary'] = "Error"
                                else:
                                    st.success(f"{status_msg} - Completed")
                                    df.at[index, 'Summary'] = result.get('summary', 'Error')
                            except Exception as e:
                                st.warning(f"{status_msg} - Can't generate summary: {str(e)}")
                                df.at[index, 'Summary'] = "Error"
                                continue
                    st.session_state.processed_df = df
                    st.session_state.processing_complete = True
                st.success("Processing complete! You can now download the file.")
            if st.session_state.processing_complete and st.session_state.processed_df is not None:
                st.subheader("Summaries from Excel File")
                df = st.session_state.processed_df
                for index, row in df.iterrows():
                    url = row.get("BillTextURL", "")
                    summary = row.get("Summary", "No summary available")
                    if pd.notna(url) and url.strip():
                        with st.expander(f"Summary for {url}", expanded=False):
                            if summary.strip() == "Error":
                                st.warning("Could not generate summary")
                            else:
                                st.subheader("Summary")
                                st.markdown(summary)
                            st.write("---")
            if st.session_state.processing_complete and st.session_state.processed_df is not None:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    st.session_state.processed_df.to_excel(writer, index=False)
                original_filename = st.session_state.prev_upload
                base_name = original_filename.rsplit('.', 1)[0]
                last_7_chars = base_name[-7:] if len(base_name) >= 7 else base_name
                processed_filename = f"{last_7_chars}_summarized.xlsx"
                st.download_button(
                    label="Download Updated Excel",
                    data=output.getvalue(),
                    file_name=processed_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key='download_btn'
                )

    # PDF file processing
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf is not None:
        st.success("PDF file uploaded successfully!")
        if st.button("Summarize PDF"):
            with st.spinner('Processing PDF...'):
                result = process_input(uploaded_pdf, custom_prompt=custom_prompt)
                logging.info(f"Result from process_input for PDF: {result}")
                if isinstance(result, dict) and "error" in result:
                    st.warning(f"Failed to process PDF: {result['error']}")
                elif isinstance(result, dict) and "summary" in result:
                    st.success("Summarization complete!")
                    st.session_state.all_summaries[uploaded_pdf.name] = {"summary": result["summary"]}
                    st.session_state.last_processed_url = uploaded_pdf.name
                    display_summary(st.session_state.all_summaries[uploaded_pdf.name], uploaded_pdf.name, use_typewriter=True)
                else:
                    st.error("An unexpected error occurred: Invalid result format.")
        if uploaded_pdf.name in st.session_state.all_summaries and uploaded_pdf.name != st.session_state.last_processed_url:
            display_summary(st.session_state.all_summaries[uploaded_pdf.name], uploaded_pdf.name)

if __name__ == "__main__":
    main()