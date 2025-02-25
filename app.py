import streamlit as st
import pandas as pd
from summarize_togetherAI import process_input
import subprocess
from io import BytesIO
import logging
import re
from PyPDF2 import PdfReader

# Configure logging at WARNING level to avoid INFO-level logs (like raw JSON outputs)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

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

def format_highlights(highlights_text):
    """Convert plain text highlights to formatted markdown for display and remove Markdown for saving."""
    # Remove markdown headers (e.g., "# Heading" or "## Subheading")
    highlights_text = re.sub(r'#{1,6}\s*', '', highlights_text)  
    
    # Replace bullet points with hyphens for consistency
    highlights_text = highlights_text.replace("•", "-").replace("—", "--")

    # Remove bold formatting (**text** -> text)
    highlights_text = re.sub(r'\*\*(.*?)\*\*', r'\1', highlights_text)  
    
    return highlights_text

def justify_text(text):
    """Wrap text in a div that justifies its alignment."""
    return f"<div style='text-align: justify;'>{text}</div>"

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file using PyPDF2 (for fallback purposes)."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Stateside Bill Summarization")
    st.write("Enter a URL of a Stateside bill to summarize its content, upload an Excel file, or upload a PDF file.")

    # Initialize session state for persistent data
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'prev_upload' not in st.session_state:
        st.session_state.prev_upload = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'all_summaries' not in st.session_state:
        st.session_state.all_summaries = {}

    # -------- Single URL Processing with realtime feedback --------
    url = st.text_input("Enter URL")
    if st.button("Summarize URL"):
        if url:
            # Create a placeholder for realtime status updates
            status_placeholder = st.empty()
            status_placeholder.info("Starting summary generation...")
            with st.spinner('Generating summary...'):
                # Call the processing function (this call will block until complete)
                result = process_input(url)
            if isinstance(result, dict) and "error" in result:
                status_placeholder.warning(f"Failed to process URL: {result['error']}")
            elif isinstance(result, dict):
                status_placeholder.success("Summarization complete!")
                # Store the summaries, formatting each with justified text for display.
                st.session_state.all_summaries[url] = {
                    "abstractive": justify_text(result["abstractive"]),
                    "extractive": justify_text(result["extractive"]),
                    "highlights": justify_text(format_highlights(result["highlights"]))
                }
            else:
                st.error("An unexpected error occurred.")
        else:
            st.error("Please enter a valid URL.")

    # Display URL summaries
    if st.session_state.all_summaries:
        st.subheader("URL Summaries")
        for url, summaries in st.session_state.all_summaries.items():
            with st.expander(f"Summary for {url}", expanded=True):
                if summaries["extractive"].strip() == "Extractive summary not found.":
                    st.warning("Could not generate extractive summary")
                else:
                    st.subheader("Extractive Summary")
                    st.markdown(summaries["extractive"], unsafe_allow_html=True)
                st.subheader("Abstractive Summary")
                st.markdown(summaries["abstractive"], unsafe_allow_html=True)
                st.subheader("Highlights & Analysis")
                st.markdown(summaries["highlights"], unsafe_allow_html=True)
                st.write("---")

    # -------- Excel File Processing with progress updates --------
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    if uploaded_file is not None:
        if st.session_state.prev_upload != uploaded_file.name:
            st.session_state.processed_df = pd.read_excel(uploaded_file)
            st.session_state.prev_upload = uploaded_file.name
            st.session_state.processing_complete = False
            st.session_state.all_summaries = {}
            for col in ['Extractive Summary', 'Abstractive Summary', 'Highlights & Analysis']:
                if col not in st.session_state.processed_df.columns:
                    st.session_state.processed_df[col] = ''
        df = st.session_state.processed_df
        if 'BillState' not in df.columns or 'BillTextURL' not in df.columns:
            st.error("File must contain 'BillState' and 'BillTextURL' columns")
        else:
            if st.button("Process Excel File"):
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                with st.spinner('Processing URLs... This may take several minutes'):
                    total_urls = len(df['BillTextURL'].dropna())
                    processed_count = 0
                    for index, row in df.iterrows():
                        url = row['BillTextURL']
                        if pd.notna(url) and df.at[index, 'Extractive Summary'] == '':
                            processed_count += 1
                            status_msg = f"Processing URL {processed_count}/{total_urls}: {url}"
                            status_placeholder.info(status_msg)
                            try:
                                result = process_input(url)
                                if isinstance(result, dict) and "error" in result:
                                    st.warning(f"{status_msg} - Can't generate summary")
                                    df.at[index, 'Extractive Summary'] = "Error"
                                    df.at[index, 'Abstractive Summary'] = "Error"
                                    df.at[index, 'Highlights & Analysis'] = "Error"
                                else:
                                    st.success(f"{status_msg} - Completed")
                                    df.at[index, 'Extractive Summary'] = result.get('extractive', 'Error')
                                    df.at[index, 'Abstractive Summary'] = result.get('abstractive', 'Error')
                                    highlights = format_highlights(result.get('highlights', 'Error'))
                                    df.at[index, 'Highlights & Analysis'] = highlights
                            except Exception as e:
                                st.warning(f"{status_msg} - Can't generate summary")
                                df.at[index, 'Extractive Summary'] = "Error"
                                df.at[index, 'Abstractive Summary'] = "Error"
                                df.at[index, 'Highlights & Analysis'] = "Error"
                            # Update the progress bar after each URL is processed
                            progress_bar.progress(int((processed_count / total_urls) * 100))
                    st.session_state.processed_df = df
                    st.session_state.processing_complete = True
                st.success("Processing complete! You can now download the file.")
            if st.session_state.processing_complete and st.session_state.processed_df is not None:
                st.subheader("Summaries from Excel File")
                df = st.session_state.processed_df
                for index, row in df.iterrows():
                    url = row.get("BillTextURL", "")
                    extractive_summary = row.get("Extractive Summary", "No summary available")
                    abstractive_summary = row.get("Abstractive Summary", "No summary available")
                    highlights = row.get("Highlights & Analysis", "No highlights available")
                    if pd.notna(url) and url.strip():
                        with st.expander(f"Summary for {url}", expanded=False):
                            if extractive_summary.strip() == "Error":
                                st.warning("Could not generate extractive summary")
                            else:
                                st.subheader("Extractive Summary")
                                st.markdown(justify_text(extractive_summary), unsafe_allow_html=True)
                            st.subheader("Abstractive Summary")
                            st.markdown(justify_text(abstractive_summary), unsafe_allow_html=True)
                            st.subheader("Highlights & Analysis")
                            st.markdown(justify_text(format_highlights(highlights)), unsafe_allow_html=True)
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

    # -------- PDF File Processing with realtime feedback --------
    uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])
    if uploaded_pdf is not None:
        st.success("PDF file uploaded successfully!")
        if st.button("Summarize PDF"):
            status_placeholder = st.empty()
            status_placeholder.info("Generating summary for the PDF...")
            with st.spinner('Processing PDF...'):
                # Pass the uploaded file object directly to process_input
                result = process_input(uploaded_pdf)
            if isinstance(result, dict) and "error" in result:
                status_placeholder.warning(f"Failed to process PDF: {result['error']}")
            elif isinstance(result, dict):
                status_placeholder.success("Summarization complete!")
                st.session_state.all_summaries[uploaded_pdf.name] = {
                    "abstractive": justify_text(result["abstractive"]),
                    "extractive": justify_text(result["extractive"]),
                    "highlights": justify_text(format_highlights(result["highlights"]))
                }
            else:
                st.error("An unexpected error occurred.")
        if uploaded_pdf.name in st.session_state.all_summaries:
            summaries = st.session_state.all_summaries[uploaded_pdf.name]
            with st.expander(f"Summary for {uploaded_pdf.name}", expanded=True):
                if summaries["extractive"].strip() == "Extractive summary not found.":
                    st.warning("Could not generate extractive summary")
                else:
                    st.subheader("Extractive Summary")
                    st.markdown(summaries["extractive"], unsafe_allow_html=True)
                st.subheader("Abstractive Summary")
                st.markdown(summaries["abstractive"], unsafe_allow_html=True)
                st.subheader("Highlights & Analysis")
                st.markdown(summaries["highlights"], unsafe_allow_html=True)
                st.write("---")

if __name__ == "__main__":
    main()
