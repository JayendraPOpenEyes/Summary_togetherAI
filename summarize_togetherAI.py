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
        if not base_name or not base_name.strip():
            raise ValueError("Cache name must be specified")
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
        pdf_bytes = pdf_content.read()
        native_text = self.extract_text_from_pdf_native(pdf_bytes)
        if native_text and not self.is_blank_text(native_text):
            logging.info("Native PDF text extraction succeeded.")
            return native_text
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
            native_text = self.extract_text_from_pdf_native(pdf_bytes)
            if native_text and not self.is_blank_text(native_text):
                return {"text": native_text, "content_type": "pdf", "error": None}
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
        """Streamlined generation of JSON structure directly from text."""
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
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)

    def generate_summaries_with_togetherai(self, combined_text, custom_prompt=None):
        combined_text = self.truncate_text(combined_text, max_tokens=4000)
        if custom_prompt:
            prompt = f"{custom_prompt}\n\nText to process:\n{combined_text}"
        else:
            prompt = f"Summarize the following text:\n{combined_text}"
        logging.info(f"Sending prompt to TogetherAI: {prompt[:100]}...")
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
            response = requests.post("https://api.together.ai/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            summary = response_data["choices"][0]["message"]["content"].strip()
            logging.info(f"Received summary: {summary[:100]}...")
            return {"summary": summary}
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return {"summary": f"Error generating summary: {str(e)}"}

    def get_hash(self, text, custom_prompt=None):
        # Combine text and custom_prompt (if provided) for the hash
        combined_input = f"{text}{custom_prompt or ''}"
        return hashlib.md5(combined_input.encode('utf-8')).hexdigest()

    def get_cache_file_path(self, base_name, text_hash):
        folder = self.get_save_directory(base_name)
        return os.path.join(folder, f"{base_name}_{text_hash}_cache.json")

    def get_cached_summary(self, text, base_name, custom_prompt=None, cache_expiry=3600):
        text_hash = self.get_hash(text, custom_prompt)
        cache_file = self.get_cache_file_path(base_name, text_hash)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if time.time() - cache.get("timestamp", 0) < cache_expiry and cache.get("text_hash") == text_hash:
                    logging.info("Returning cached summary.")
                    return cache.get("summary")
            except Exception as e:
                logging.error(f"Error reading cache: {str(e)}")
        return None

    def update_cached_summary(self, text, summary, base_name, custom_prompt=None):
        text_hash = self.get_hash(text, custom_prompt)
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
            logging.error(f"Error writing cache: {str(e)}")

    def process_raw_text(self, text, base_name="raw_text", custom_prompt=None):
        clean_text = self.preprocess_text(text)
        cached_summary = self.get_cached_summary(clean_text, base_name, custom_prompt)
        if cached_summary:
            return cached_summary
        summary = self.generate_summaries_with_togetherai(clean_text, custom_prompt)
        self.process_full_text_to_json(clean_text, base_name)
        self.update_cached_summary(clean_text, summary, base_name, custom_prompt)
        return {"model": self.model, "summary": summary["summary"]}

def process_input(input_data, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", custom_prompt=None):
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

        cached_summary = processor.get_cached_summary(clean_text, base_name, custom_prompt)
        if cached_summary:
            return cached_summary

        summary = processor.generate_summaries_with_togetherai(clean_text, custom_prompt)
        processor.process_full_text_to_json(clean_text, base_name)
        processor.update_cached_summary(clean_text, summary, base_name, custom_prompt)
        return {"model": model, "summary": summary["summary"]}
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}")
        return {"error": f"An error occurred: {str(e)}", "model": model}