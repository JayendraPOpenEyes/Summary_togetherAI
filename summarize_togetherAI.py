import os
import re
import io
import json
import time
import hashlib
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

import requests
import pdfkit
import tiktoken
from bs4 import BeautifulSoup
from PIL import Image
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import pytesseract  # For OCR
import fitz  # PyMuPDF for native PDF text extraction

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base directory to save all data
SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)

class TextProcessor:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.together_api_key = os.getenv('TOGETHERAI_API_KEY')
        if not self.together_api_key:
            raise ValueError("TogetherAI API key is missing. Ensure the TOGETHERAI_API_KEY is set in the .env file.")
        self.model = model

    def get_save_directory(self, base_name):
        folder_path = os.path.join(SAVE_DIR, base_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def get_base_name_from_link(self, link):
        parts = link.split('/')
        meaningful_parts = [part for part in parts[-4:] if part and part.lower() not in ['pdf', 'html', 'htm']]
        base_name = '_'.join(meaningful_parts) or '_'.join(parts)
        base_name = re.sub(r"\.(htm|html|pdf)$", "", base_name, flags=re.IGNORECASE)
        base_name = re.sub(r"[^\w\-_\. ]", "_", base_name)
        if len(base_name) > 50:
            base_name = base_name[:50]
        return base_name or "default_name"

    def is_google_cache_link(self, link):
        return "webcache.googleusercontent.com" in link

    def is_blank_text(self, text):
        clean_text = re.sub(r"\s+", "", text).strip()
        return len(clean_text) < 100

    def process_image_with_tesseract(self, image_path):
        try:
            return pytesseract.image_to_string(Image.open(image_path))
        except Exception as e:
            logging.error(f"Error processing image with Tesseract: {str(e)}")
            return ""

    def extract_text_from_pdf_native(self, pdf_bytes):
        """Try to extract text natively using PyMuPDF."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text.strip()
        except Exception as e:
            logging.error("Error in native PDF extraction: " + str(e))
            return ""

    def extract_text_from_pdf(self, pdf_content, link):
        base_name = self.get_base_name_from_link(link)
        folder = self.get_save_directory(base_name)
        # Read PDF bytes once so we can reuse them
        pdf_bytes = pdf_content.read()
        # Attempt native extraction first
        native_text = self.extract_text_from_pdf_native(pdf_bytes)
        if native_text and not self.is_blank_text(native_text):
            logging.info("Native PDF text extraction succeeded.")
            return native_text
        # Fallback to OCR using in-memory images
        images = convert_from_bytes(pdf_bytes)
        logging.info(f"OCR fallback: converting {len(images)} pages to images.")
        combined_text = ""

        def process_page(i, img):
            img_filename = f"{base_name}_page_{i+1}.png"
            img_path = os.path.join(folder, img_filename)
            img.save(img_path, 'PNG')
            logging.info(f"Saved image: {img_path}")
            return self.process_image_with_tesseract(img_path)

        with ThreadPoolExecutor() as executor:
            results = executor.map(lambda x: process_page(x[0], x[1]), enumerate(images))
            for text in results:
                combined_text += text + "\n"
        return combined_text

    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        return soup.get_text(separator=' ').strip()

    async def async_extract_text_from_url(self, url):
        """Asynchronously fetches and processes a URL."""
        if self.is_google_cache_link(url):
            return {"text": "", "content_type": None, "error": "google_cache"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return {"text": "", "content_type": None, "error": f"HTTP error {response.status}"}
                    content_type = response.headers.get('Content-Type', '').lower()
                    content = await response.read()
                    base_name = self.get_base_name_from_link(url)
                    folder = self.get_save_directory(base_name)
                    if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
                        pdf_path = os.path.join(folder, f"{base_name}.pdf")
                        with open(pdf_path, 'wb') as f:
                            f.write(content)
                        logging.info(f"Saved PDF: {pdf_path}")
                        text = self.extract_text_from_pdf(io.BytesIO(content), url)
                        if self.is_blank_text(text):
                            return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
                        return {"text": text, "content_type": "pdf", "error": None}
                    elif url.lower().endswith(('.htm', '.html')) or 'text/html' in content_type:
                        html_path = os.path.join(folder, f"{base_name}.html")
                        with open(html_path, 'wb') as f:
                            f.write(content)
                        logging.info(f"Saved HTML: {html_path}")
                        text = self.extract_text_from_html(content)
                        return {"text": text, "content_type": "html", "error": None}
                    else:
                        return {"text": "", "content_type": None, "error": "unsupported_type"}
        except Exception as e:
            logging.error(f"Error fetching URL {url}: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def process_uploaded_pdf(self, pdf_file, base_name="uploaded_pdf"):
        try:
            folder = self.get_save_directory(base_name)
            pdf_path = os.path.join(folder, f"{base_name}.pdf")
            pdf_bytes = pdf_file.read()
            if not pdf_bytes:
                return {"text": "", "content_type": "pdf", "error": "Empty PDF file"}
            with open(pdf_path, 'wb') as f:
                f.write(pdf_bytes)
            logging.info(f"Saved uploaded PDF: {pdf_path}")
            # Attempt native extraction first; fallback to OCR if needed
            native_text = self.extract_text_from_pdf_native(pdf_bytes)
            if native_text and not self.is_blank_text(native_text):
                return {"text": native_text, "content_type": "pdf", "error": None}
            # If native extraction fails, use OCR in parallel
            images = convert_from_bytes(pdf_bytes)
            logging.info(f"OCR fallback: converting {len(images)} page(s) to images.")
            combined_text = ""

            def process_page(i, img):
                img_filename = f"{base_name}_page_{i+1}.png"
                img_path = os.path.join(folder, img_filename)
                img.save(img_path, 'PNG')
                logging.info(f"Saved image: {img_path}")
                return self.process_image_with_tesseract(img_path)

            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda x: process_page(x[0], x[1]), enumerate(images))
                for text in results:
                    combined_text += text + "\n"
            if self.is_blank_text(combined_text):
                return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
            return {"text": combined_text, "content_type": "pdf", "error": None}
        except Exception as e:
            logging.error(f"Error processing uploaded PDF: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def process_uploaded_html(self, html_file, base_name="uploaded_html"):
        try:
            folder = self.get_save_directory(base_name)
            html_path = os.path.join(folder, f"{base_name}.html")
            html_bytes = html_file.read()
            if not html_bytes:
                return {"text": "", "content_type": "html", "error": "Empty HTML file"}
            with open(html_path, 'wb') as f:
                f.write(html_bytes)
            logging.info(f"Saved uploaded HTML: {html_path}")
            text = self.extract_text_from_html(html_bytes)
            return {"text": text, "content_type": "html", "error": None}
        except Exception as e:
            logging.error(f"Error processing uploaded HTML: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def preprocess_text(self, text):
        text = re.sub(r"[\r\n]{2,}", "\n", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def generate_structured_json(self, text):
        """
        Streamlined generation of JSON structure directly from text.
        Splits the text into paragraphs. Paragraphs with >10 words are added under 'p',
        otherwise they are added under 'h1'.
        """
        paragraphs = text.split('\n')
        json_data = {"h1": [], "p": []}
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para.split()) > 10:
                json_data["p"].append(para)
            else:
                json_data["h1"].append(para)
        return json_data

    def process_full_text_to_json(self, text, base_name):
        json_data = self.generate_structured_json(text)
        base_folder = self.get_save_directory(base_name)
        json_path = os.path.join(base_folder, f"{base_name}.json")
        try:
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            logging.info(f"Saved JSON: {json_path}")
        except Exception as e:
            logging.error(f"Error saving JSON: {str(e)}")
        return json_data

    def truncate_text(self, text, max_tokens=3000):
        # Explicitly get an encoding as the model "llama" is not automatically mapped.
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)

    def generate_summaries_with_chatgpt(self, combined_text):
        combined_text = self.truncate_text(combined_text, max_tokens=4000)
        prompt = f"""
Generate the following summaries for the text below. Please adhere to these instructions:

For Abstractive Summary:
- The summary should be concise and not very long.
- It should cover all the key points very shortly.
- Summarize the content in one short paragraph (maximum 8 sentences).

For Extractive Summary:
- Generate a minimum of 2 paragraphs if the content is sufficiently long; adjust accordingly if the content is short.
- Provide a sensible extractive summary capturing the main ideas.

For Highlights & Analysis:
- Produce 15 to 20 bullet points grouped under 4 meaningful headings.
- Each heading should be relevant to the content and include bullet points with key details.
- Highlights should be in the form of headings only, followed by bullet points.

Use the following markers exactly for each section:

Abstractive Summary:
[Abstractive]

Extractive Summary:
[Extractive]

Highlights & Analysis:
[Highlights]

Only output the text within these markers without any additional commentary.

Text:
{combined_text}
"""
        try:
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1500,
            }
            # Updated endpoint URL to include the v1 prefix.
            response = requests.post("https://api.together.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            summaries = response_data["choices"][0]["message"]["content"]
            abstractive_match = re.search(r"\[Abstractive\](.*?)\[Extractive\]", summaries, re.DOTALL)
            extractive_match = re.search(r"\[Extractive\](.*?)\[Highlights\]", summaries, re.DOTALL)
            highlights_match = re.search(r"\[Highlights\](.*)", summaries, re.DOTALL)
            return {
                "extractive": extractive_match.group(1).strip() if extractive_match else "Extractive summary not found.",
                "abstractive": abstractive_match.group(1).strip() if abstractive_match else "Abstractive summary not found.",
                "highlights": highlights_match.group(1).strip() if highlights_match else "Highlights not found."
            }
        except Exception as e:
            logging.error(f"Error generating summaries: {str(e)}")
            return {
                "extractive": "Error generating extractive summary.",
                "abstractive": "Error generating abstractive summary.",
                "highlights": "Error generating highlights."
            }

    # Caching methods based on content hash
    def get_hash(self, text):
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_cache_file_path(self, base_name, text_hash):
        folder = self.get_save_directory(base_name)
        return os.path.join(folder, f"summary_cache_{text_hash}.json")

    def get_cached_summary(self, text, base_name, cache_expiry=3600):
        text_hash = self.get_hash(text)
        cache_file = self.get_cache_file_path(base_name, text_hash)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if time.time() - cache.get("timestamp", 0) < cache_expiry:
                    logging.info("Returning cached summary.")
                    return cache.get("summary")
            except Exception as e:
                logging.error("Error reading cache: " + str(e))
        return None

    def update_cached_summary(self, text, summary, base_name):
        text_hash = self.get_hash(text)
        cache_file = self.get_cache_file_path(base_name, text_hash)
        cache = {
            "text_hash": text_hash,
            "summary": summary,
            "timestamp": time.time()
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            logging.info("Cache updated.")
        except Exception as e:
            logging.error("Error writing cache: " + str(e))

    def process_raw_text(self, text, base_name="raw_text"):
        clean_text = self.preprocess_text(text)
        # Check for cached summary first
        cached_summary = self.get_cached_summary(clean_text, base_name)
        if cached_summary:
            return cached_summary
        summaries = self.generate_summaries_with_chatgpt(clean_text)
        self.process_full_text_to_json(clean_text, base_name)
        # Update cache with new summary
        self.update_cached_summary(clean_text, summaries, base_name)
        return {
            "model": self.model,
            "extractive": summaries["extractive"],
            "abstractive": summaries["abstractive"],
            "highlights": summaries["highlights"]
        }

def process_input(input_data, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    try:
        processor = TextProcessor(model=model)
        if hasattr(input_data, "read") and not isinstance(input_data, str):
            file_identifier = input_data.name if hasattr(input_data, "name") else "uploaded_file"
            logging.info(f"Processing uploaded file: {file_identifier}")
            _, ext = os.path.splitext(file_identifier)
            ext = ext.lower()
            if ext in [".htm", ".html"]:
                result = processor.process_uploaded_html(
                    input_data, 
                    base_name=file_identifier[-7:] if len(file_identifier) >= 7 else file_identifier
                )
            elif ext == ".pdf":
                result = processor.process_uploaded_pdf(
                    input_data, 
                    base_name=file_identifier[-7:] if len(file_identifier) >= 7 else file_identifier
                )
            else:
                result = {"text": input_data.read(), "content_type": "raw", "error": None}
            if result["error"]:
                return {"error": result["error"], "model": model}
            clean_text = processor.preprocess_text(result["text"])
            base_name = file_identifier[-7:] if len(file_identifier) >= 7 else file_identifier
        elif isinstance(input_data, str) and input_data.startswith(("http://", "https://")):
            # Use asynchronous URL fetching
            result = asyncio.run(processor.async_extract_text_from_url(input_data))
            if result["error"]:
                return {"error": result["error"], "model": model}
            clean_text = processor.preprocess_text(result["text"])
            base_name = processor.get_base_name_from_link(input_data)
        elif isinstance(input_data, str):
            clean_text = processor.preprocess_text(input_data)
            base_name = "raw_text"
        else:
            return {"error": "Invalid input type. Expected URL, raw text, or an uploaded file.", "model": model}

        # Check cache for summaries to avoid unnecessary API calls if content hasn't changed
        cached_summary = processor.get_cached_summary(clean_text, base_name)
        if cached_summary:
            return cached_summary

        summaries = processor.generate_summaries_with_chatgpt(clean_text)
        processor.process_full_text_to_json(clean_text, base_name)
        # Update cache with new summary
        processor.update_cached_summary(clean_text, summaries, base_name)
        return {
            "model": model,
            "extractive": summaries["extractive"],
            "abstractive": summaries["abstractive"],
            "highlights": summaries["highlights"]
        }
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}")
        return {"error": f"An error occurred: {str(e)}", "model": model}
