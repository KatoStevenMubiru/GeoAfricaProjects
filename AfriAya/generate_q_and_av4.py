# -*- coding: utf-8 -*-
import os
import pandas as pd
from openai import OpenAI # Use the modern OpenAI library
import json
from PIL import Image
import io
import time
import base64
import re
import mimetypes
import concurrent.futures # Added for parallelism
from tqdm import tqdm # Added for progress bar
import traceback # For detailed error logging

# --- Add google-genai Imports ---
# pip install google-genai
try:
    from google import genai
    # import google.api_core.exceptions
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    print("Warning: google-genai library not found. Translation step will be skipped.")
    print("Install using: pip install google-genai")
    GEMINI_SDK_AVAILABLE = False


# --- Configuration ---
PARENT_OUTPUT_DIR = "BingImages" # Source directory
# --- API Keys (Load securely!) ---
COHERE_API_KEY = ""
GOOGLE_API_KEY = ""
# --- Output ---
OUTPUT_FILE = "image_captions_qa_translated_parallel_batch.json" # Updated output filename
# --- Processing Controls ---
MAX_WORKERS = 8 # Number of parallel API connections (Adjust based on your system/API limits)
ENABLE_TRANSLATION = False # Set to False to disable translation step
# --- Models ---
VISION_MODEL_ID = "c4ai-aya-vision-32b" # Cohere model via OpenAI compat endpoint
GEMINI_MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Translation model
# --- API Settings ---
API_CALL_DELAY_SECONDS = 0 # Reduced delay, parallelism handles some throttling
MAX_RETRIES = 2
RETRY_DELAY = 3
# --- Image Settings ---
TARGET_MAX_SIZE_BYTES = 4.8 * 1024 * 1024
MIN_JPEG_QUALITY = 50
# --- Q&A Types ---
QUESTION_TYPES_TO_GENERATE = [
    "Descriptive", "Contextual True/False",
    "Contextual Multiple Choice", "Object/Action Identification"
]
# --- Gemini Settings ---
# GEMINI_TEMPERATURE = 0.2
GEMINI_SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in
    ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
     "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
]
# GEMINI_GENERATION_CONFIG = genai.types.GenerationConfig(temperature=GEMINI_TEMPERATURE) if GEMINI_SDK_AVAILABLE else None

# --- Culture to Language Mapping ---
ORIGINAL_CULTURES = [
    "Swahili", "Igbo", "Hausa", "Kinyarwanda", "Yoruba",
    "Arabic Darija (Moroccan dialect)", "Zulu", "Luganda (Ganda)",
    "Nyankore", "Gishu", "Chichewa", "Twi", "Urhobo", "Somali"
]
def sanitize_name_for_directory(name):
    safe_name = re.sub(r'[^\w\-]+', '_', name).lower().strip('_')
    return f"{safe_name or 'unknown_culture'}_images"
CULTURE_DIR_TO_LANGUAGE_MAP = {sanitize_name_for_directory(c): c for c in ORIGINAL_CULTURES}
CULTURE_DIR_TO_LANGUAGE_MAP.setdefault("unknown_culture_images", "English")

# --- Initialize API Clients ---
client_openai_compat = None
gemini_model = None

def initialize_clients():
    """Initializes API clients and returns them."""
    global client_openai_compat, gemini_model # Allow modification of globals
    # Cohere Client (via OpenAI Compat)
    if not COHERE_API_KEY or COHERE_API_KEY == "<YOUR_COHERE_API_KEY>":
        print("Error: COHERE_API_KEY not set."); return None, None
    try:
        client_openai_compat = OpenAI(api_key=COHERE_API_KEY, base_url="https://api.cohere.ai/compatibility/v1")
        print("OpenAI client for Cohere compatibility initialized.")
    except Exception as e:
        print(f"Error initializing OpenAI client for Cohere compatibility: {e}"); return None, None

    # Gemini Client
    if GEMINI_SDK_AVAILABLE:
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "<YOUR_GOOGLE_API_KEY>":
            print("Warning: GOOGLE_API_KEY not set. Translation will be skipped.")
        else:
            try:
                genai.configure(api_key=GOOGLE_API_KEY)
                gemini_model = genai.GenerativeModel(
                    GEMINI_MODEL_NAME,
                    safety_settings=GEMINI_SAFETY_SETTINGS,
                    generation_config=GEMINI_GENERATION_CONFIG
                )
                print(f"Gemini model ({GEMINI_MODEL_NAME}) initialized via google-genai.")
            except Exception as e:
                print(f"Error initializing google-genai client/model: {e}. Translation skipped.")
                gemini_model = None # Ensure it's None if init fails
    else:
         print("google-genai SDK not available. Translation skipped.")

    return client_openai_compat, gemini_model

# --- Helper Function for Image Compression ---
def compress_image_if_needed(image_path, target_max_size, min_jpeg_quality):
    """Reads image, compresses if needed, and returns bytes."""
    try:
        with open(image_path, "rb") as img_file: image_bytes = img_file.read()
    except Exception as read_e: print(f"      Error reading image file {image_path}: {read_e}"); return None
    if len(image_bytes) <= target_max_size: return image_bytes
    # print(f"      Compressing {os.path.basename(image_path)}...") # Less verbose
    try:
        img = Image.open(io.BytesIO(image_bytes)); img_format = getattr(img, 'format', 'JPEG').upper()
        if img.mode in ('RGBA', 'P') and img_format != 'PNG': img = img.convert('RGB'); img_format = 'JPEG'
        output_buffer = io.BytesIO()
        if img_format == 'JPEG':
            quality = 85
            while quality >= min_jpeg_quality:
                output_buffer.seek(0); output_buffer.truncate()
                img.save(output_buffer, format='JPEG', quality=quality, optimize=True)
                if output_buffer.tell() <= target_max_size: return output_buffer.getvalue()
                quality -= 10
            # print(f"      Warning: JPEG couldn't reach target size @ quality {min_jpeg_quality}.")
            return output_buffer.getvalue()
        elif img_format == 'PNG':
             output_buffer.seek(0); output_buffer.truncate(); img.save(output_buffer, format='PNG', optimize=True)
             if output_buffer.tell() <= target_max_size: return output_buffer.getvalue()
             else:
                 # print("        Optimized PNG too large. Converting to JPEG..."); # Less verbose
                 if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
                 output_buffer.seek(0); output_buffer.truncate(); img.save(output_buffer, format='JPEG', quality=75, optimize=True)
                 if output_buffer.tell() <= target_max_size: return output_buffer.getvalue()
                 else: print(f"      Warning: PNG->JPEG still exceeds target size for {os.path.basename(image_path)}."); return output_buffer.getvalue()
        else: # Handle other formats
            # print(f"        Unsupported format '{img_format}'. Converting to JPEG..."); # Less verbose
            if img.mode in ('RGBA', 'P'): img = img.convert('RGB')
            output_buffer.seek(0); output_buffer.truncate(); img.save(output_buffer, format='JPEG', quality=75, optimize=True)
            if output_buffer.tell() <= target_max_size: return output_buffer.getvalue()
            else: print(f"      Warning: Conversion of {img_format}->JPEG still exceeds target size for {os.path.basename(image_path)}."); return output_buffer.getvalue()
    except Exception as compress_e: print(f"      Error during image compression for {os.path.basename(image_path)}: {compress_e}."); return image_bytes # Return original on error

# --- Helper Function for Vision Call (OpenAI Compatibility Endpoint) ---
def get_vision_response_openai_compat(client, image_path, text_prompt):
    """Sends image (compressed) and prompt using OpenAI SDK to Cohere's compatibility endpoint."""
    raw_response_text = None; parsed_json = None
    compressed_image_bytes = compress_image_if_needed(image_path, TARGET_MAX_SIZE_BYTES, MIN_JPEG_QUALITY)
    if compressed_image_bytes is None: return None, "Error: Could not read or compress image."
    try:
        image_base64 = base64.b64encode(compressed_image_bytes).decode('utf-8')
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            try: img_format = Image.open(io.BytesIO(compressed_image_bytes)).format; mime_type = Image.MIME.get(img_format.upper()) if img_format else None
            except Exception: pass
        if not mime_type: mime_type = 'image/jpeg'
        base64_data_url = f"data:{mime_type};base64,{image_base64}"
    except Exception as b64_e: return None, f"Error: Could not create base64 data URL - {b64_e}"

    for attempt in range(MAX_RETRIES):
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}, {"type": "image_url", "image_url": {"url": base64_data_url}}]}]
            response = client.chat.completions.create(model=VISION_MODEL_ID, messages=messages, temperature=0.3, max_tokens=1024)
            if response.choices and response.choices[0].message: raw_response_text = response.choices[0].message.content.strip()
            else: raw_response_text = str(response); print("Warning: Could not extract message content from OpenAI response.")
            try:
                # Attempt to parse JSON strictly
                json_start = raw_response_text.find('{'); json_end = raw_response_text.rfind('}') + 1
                if json_start != -1 and json_end != -1: parsed_json = json.loads(raw_response_text[json_start:json_end])
                # else: print("Warning: Could not find JSON block in OpenAI Compat response.") # Less verbose
            except json.JSONDecodeError: pass # Ignore parse error here, handle later
            except Exception as parse_e: print(f"Warning: Unexpected error during OpenAI Compat JSON parsing: {parse_e}")
            break # Success or parse error, break retry
        except Exception as e:
            print(f"Error calling OpenAI Compat Endpoint (Attempt {attempt + 1}/{MAX_RETRIES}) for {os.path.basename(image_path)}: {type(e).__name__} - {e}")
            raw_response_text = f"Error: {e}"
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY * (2**attempt)) # Exponential backoff
            else: print(f"Max retries reached for OpenAI Compat call on {os.path.basename(image_path)}.")
    time.sleep(API_CALL_DELAY_SECONDS)
    return parsed_json, raw_response_text

# --- Helper Function for BATCH Translation Call (google-genai) --- NEW
def translate_batch_gemini_native(texts_to_translate, target_language, model):
    """
    Translates a batch of texts using the initialized google-genai model.
    Expects and attempts to parse a JSON list response.
    Returns a list of translated texts (or None for failed items), matching input order.
    """
    if not model or not texts_to_translate:
        return [None] * len(texts_to_translate or []) # Return list of Nones

    # Create a numbered list string for the prompt
    numbered_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts_to_translate)])

    # Construct the prompt asking for a JSON list output
    prompt = f"""Translate the following {len(texts_to_translate)} English text items to {target_language}.
Please provide the translations in the EXACT same order as the input items.
Format your response STRICTLY as a single JSON list of strings, where each string is the translation of the corresponding input item. Do not include the original numbers or any other text outside the JSON list.

Input English Texts:
{numbered_texts}

Output JSON list of {target_language} translations:
"""

    translated_texts = [None] * len(texts_to_translate) # Initialize with Nones

    for attempt in range(MAX_RETRIES):
        try:
            # Use generate_content for non-streaming response
            response = model.generate_content(prompt)
            raw_response_text = response.text.strip()

            # Attempt to parse the raw text as JSON list
            parsed_json = None
            try:
                json_start = raw_response_text.find('['); json_end = raw_response_text.rfind(']') + 1
                if json_start != -1 and json_end != -1:
                    json_str = raw_response_text[json_start:json_end]
                    parsed_json = json.loads(json_str)
                # else: print(f"      Warning: Could not find JSON list in Gemini batch response.") # Less verbose

                if isinstance(parsed_json, list) and len(parsed_json) == len(texts_to_translate):
                    # Successfully parsed a list of the correct length
                    print(f"      Successfully parsed batch translation response ({len(parsed_json)} items).")
                    # Basic cleaning (remove potential surrounding quotes from each item)
                    translated_texts = [str(t).strip().strip('"') if t is not None else None for t in parsed_json]
                    return translated_texts # Success!
                elif isinstance(parsed_json, list):
                     print(f"      Warning: Gemini batch response list length mismatch (Expected {len(texts_to_translate)}, Got {len(parsed_json)}).")
                     # Attempt partial assignment if lengths differ but response is list
                     for i in range(min(len(parsed_json), len(texts_to_translate))):
                          translated_texts[i] = str(parsed_json[i]).strip().strip('"') if parsed_json[i] is not None else None
                     return translated_texts # Return partially filled list
                else:
                    print(f"      Warning: Gemini batch response was not a JSON list. Raw: {raw_response_text[:100]}...")

            except json.JSONDecodeError as json_e:
                print(f"      Warning: Could not decode JSON list from Gemini batch response: {json_e}")
                print(f"      Raw Gemini response was: {raw_response_text[:200]}...")
            except Exception as parse_e:
                 print(f"      Warning: Unexpected error during Gemini JSON list parsing: {parse_e}")

            # If parsing failed or list length mismatch, break retry (prompting issue likely)
            print("      Failed to get correctly formatted batch translation. Returning Nones.")
            return translated_texts # Return list potentially filled with Nones

        # except google.api_core.exceptions.ResourceExhausted as ree:
        #      print(f"      Error during Gemini batch translation (Attempt {attempt+1}): Rate limit likely exceeded - {ree}")
        #      wait_time = RETRY_DELAY * (2**attempt); time.sleep(wait_time)
        except Exception as e:
            print(f"      Error during Gemini batch translation (Attempt {attempt+1}): {type(e).__name__} - {e}")
            if attempt < MAX_RETRIES - 1: time.sleep(RETRY_DELAY)
            else: print("        Max retries reached for batch translation."); return translated_texts # Return list with Nones
    return translated_texts # Fallback return

# --- Function to process a single image ---
def process_single_image(args):
    """Processes one image: metadata, caption, Q&A, translation (batch for Q&A)."""
    image_full_path, culture_dir_path, df_metadata, target_language, should_translate, client_openai, model_gemini = args
    basename = os.path.basename(image_full_path) # For logging

    try:
        # 1. Retrieve metadata
        relative_image_path_csv = os.path.relpath(image_full_path, culture_dir_path).replace('\\', '/')
        image_meta = df_metadata[df_metadata['Relative_Path'] == relative_image_path_csv]
        query_context = image_meta['Query'].iloc[0] if not image_meta.empty else "Unknown"
        original_url = image_meta['Original_URL'].iloc[0] if not image_meta.empty else "Unknown"

        # Initialize result structure
        image_result = {
            "culture": target_language, "image_path": image_full_path,
            "original_query": query_context, "original_url": original_url,
            "caption": None, "translated_caption": None,
            "raw_caption_response": None, "qa_pairs": [], "raw_qa_responses": []
        }

        # 2. Generate Caption
        caption_prompt = """Describe this image... Format STRICTLY as JSON... Example: {"caption": "..."}"""
        parsed_caption_json, raw_caption_response = get_vision_response_openai_compat(client_openai, image_full_path, caption_prompt)
        image_result["raw_caption_response"] = raw_caption_response
        english_caption = None
        if parsed_caption_json and isinstance(parsed_caption_json, dict) and "caption" in parsed_caption_json:
            english_caption = parsed_caption_json["caption"]
            image_result["caption"] = english_caption
        else:
            print(f"Warning: Failed to parse caption JSON for {basename}")

        # 3. Translate Caption (Still done individually)
        if should_translate and english_caption:
            # print(f"        Translating caption for {basename}...") # Less verbose
            # Use the old single translation function for the caption
            image_result["translated_caption"] = translate_text_gemini_native(english_caption, target_language, model_gemini)
            time.sleep(API_CALL_DELAY_SECONDS) # Keep delay after caption translation


        # 4. Generate Q&A (English first)
        english_qa_list = [] # Store {"type": ..., "question": ..., "answer": ..., "options": ...}
        for qa_type in QUESTION_TYPES_TO_GENERATE:
            qa_prompt = f"Context: Query '{query_context}'.\n\n"
            json_structure_example = ""
            # ... (Build prompts and examples as before) ...
            if qa_type == "Descriptive": qa_prompt += "...ask question... provide answer."; json_structure_example = '{"question": "...", "answer": "..."}'
            elif qa_type == "Contextual True/False": qa_prompt += "...True/False question... Provide answer."; json_structure_example = '{"question": "True or False: ...?", "answer": "True"}'
            elif qa_type == "Contextual Multiple Choice": qa_prompt += f"...multiple-choice... 3 options... question, options list, answer."; json_structure_example = '{"question": "...", "options": ["A) ...", "B) ...", "C) ..."], "answer": "B) ..."}'
            elif qa_type == "Object/Action Identification": qa_prompt += "...ask 'What is/are...'... identify object/person/action... provide answer."; json_structure_example = '{"question": "What action ...?", "answer": "They are performing [action]."}'
            qa_prompt += f"\n\nFormat STRICTLY as JSON:\n{json_structure_example}"

            parsed_qa_json, raw_qa_response = get_vision_response_openai_compat(client_openai, image_full_path, qa_prompt)
            image_result["raw_qa_responses"].append({"type": qa_type, "raw_response": raw_qa_response})

            english_question, english_answer, options = None, None, None
            if parsed_qa_json and isinstance(parsed_qa_json, dict):
                english_question = parsed_qa_json.get("question"); english_answer = parsed_qa_json.get("answer")
                if qa_type == "Contextual Multiple Choice": options = parsed_qa_json.get("options")
                valid_qa = bool(english_question and english_answer)
                if qa_type == "Contextual Multiple Choice" and not (options and isinstance(options, list) and len(options) == 3): valid_qa = False
                if valid_qa:
                     # Store English Q&A temporarily
                     temp_qa_entry = {"type": qa_type, "question": english_question, "answer": english_answer}
                     if options: temp_qa_entry["options"] = options
                     english_qa_list.append(temp_qa_entry)
                else: print(f"Warning: Failed to parse valid Q&A JSON for {qa_type} on {basename}")
            else: print(f"Warning: Failed to get/parse Q&A JSON for {qa_type} on {basename}")

        # 5. Batch Translate Q&A if needed
        if should_translate and english_qa_list:
            print(f"        Batch translating {len(english_qa_list)*2} Q&A texts for {basename}...")
            texts_to_translate = []
            # Create flat list of Qs and As in order
            for qa in english_qa_list:
                texts_to_translate.append(qa["question"])
                texts_to_translate.append(qa["answer"])

            # Call batch translation function
            translated_texts = translate_batch_gemini_native(texts_to_translate, target_language, model_gemini)
            time.sleep(API_CALL_DELAY_SECONDS) # Delay after batch translation call

            # Assign translated texts back to the final structure
            translation_index = 0
            for i in range(len(english_qa_list)):
                 # Create final entry structure
                 final_qa_entry = {
                     "type": english_qa_list[i]["type"],
                     "question": english_qa_list[i]["question"],
                     "answer": english_qa_list[i]["answer"],
                     "translated_question": None, # Initialize
                     "translated_answer": None    # Initialize
                 }
                 if "options" in english_qa_list[i]:
                      final_qa_entry["options"] = english_qa_list[i]["options"]

                 # Assign translations if available and index is valid
                 if translated_texts:
                     if translation_index < len(translated_texts):
                         final_qa_entry["translated_question"] = translated_texts[translation_index]
                     else: print(f"Warning: Index out of bounds for translated question {translation_index} on {basename}")
                     translation_index += 1

                     if translation_index < len(translated_texts):
                          final_qa_entry["translated_answer"] = translated_texts[translation_index]
                     else: print(f"Warning: Index out of bounds for translated answer {translation_index} on {basename}")
                     translation_index += 1
                 else:
                      print(f"Warning: Batch translation failed or returned None for {basename}. Leaving translations as None.")

                 image_result["qa_pairs"].append(final_qa_entry) # Add to final results

        else:
             # If not translating, just add the English Q&A pairs
             for qa in english_qa_list:
                  final_qa_entry = {
                      "type": qa["type"],
                      "question": qa["question"],
                      "answer": qa["answer"],
                      "translated_question": None,
                      "translated_answer": None
                  }
                  if "options" in qa: final_qa_entry["options"] = qa["options"]
                  image_result["qa_pairs"].append(final_qa_entry)


        return image_result # Return the completed dictionary for this image

    except Exception as e:
        print(f"!!! Unhandled Error processing image {image_full_path}: {e}")
        traceback.print_exc() # Print full traceback for debugging worker errors
        return None # Indicate failure for this image

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Image Processing ---")
    print(f"Max Workers: {MAX_WORKERS}")
    print(f"Translation Enabled: {ENABLE_TRANSLATION}")
    print(f"Output File: {OUTPUT_FILE}")

    # Initialize clients once
    client_openai_compat, gemini_model = initialize_clients()
    if not client_openai_compat:
        print("Exiting due to OpenAI/Cohere client initialization failure.")
        exit()

    # --- Load Existing Results for Resumption ---
    processed_image_paths = set()
    existing_results = []
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing results from {OUTPUT_FILE} to resume...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            for item in existing_results:
                if isinstance(item, dict) and "image_path" in item:
                    processed_image_paths.add(item["image_path"])
            print(f"Loaded {len(existing_results)} existing results. Found {len(processed_image_paths)} processed image paths.")
        except json.JSONDecodeError: print(f"Warning: Could not decode {OUTPUT_FILE}. Starting fresh."); existing_results = []; processed_image_paths = set()
        except Exception as e: print(f"Warning: Error loading existing results: {e}. Starting fresh."); existing_results = []; processed_image_paths = set()

    # --- Gather All Image Paths and Metadata ---
    all_image_tasks = []
    print("Scanning for images and loading metadata...")
    if not os.path.isdir(PARENT_OUTPUT_DIR): print(f"Error: Parent directory '{PARENT_OUTPUT_DIR}' not found."); exit()

    for culture_dir_name in os.listdir(PARENT_OUTPUT_DIR):
        culture_dir_path = os.path.join(PARENT_OUTPUT_DIR, culture_dir_name)
        if not os.path.isdir(culture_dir_path): continue

        target_language = CULTURE_DIR_TO_LANGUAGE_MAP.get(culture_dir_name, "English")
        # Determine if translation should happen for this specific culture folder
        current_should_translate = ENABLE_TRANSLATION and GEMINI_SDK_AVAILABLE and target_language != "English" and gemini_model is not None

        # Load CSV for this culture
        culture_name_for_csv = target_language
        safe_culture_name_for_csv = re.sub(r'[^\w\-]+', '_', culture_name_for_csv).lower().strip('_') or "unknown_culture"
        csv_filename = f"{safe_culture_name_for_csv}_download_log.csv"
        csv_filepath = os.path.join(culture_dir_path, csv_filename)
        if not os.path.isfile(csv_filepath): continue
        try: df_metadata = pd.read_csv(csv_filepath)
        except Exception: continue

        # Find images in subdirs
        for root, _, files in os.walk(culture_dir_path):
            if root == culture_dir_path: continue
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
                    image_full_path = os.path.join(root, filename)
                    if image_full_path in processed_image_paths: continue # Skip if already processed
                    all_image_tasks.append((
                        image_full_path, culture_dir_path, df_metadata.copy(), # Pass copy of df slice? Or whole df? Pass whole df is simpler.
                        target_language, current_should_translate,
                        client_openai_compat, gemini_model
                    ))

    print(f"Found {len(all_image_tasks)} new images to process.")

    # --- Process Images in Parallel ---
    new_results = []
    if all_image_tasks:
        print(f"Starting parallel processing with {MAX_WORKERS} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iterator = executor.map(process_single_image, all_image_tasks)
            for result in tqdm(results_iterator, total=len(all_image_tasks), desc="Processing Images"):
                if result: new_results.append(result)
        print(f"Finished parallel processing. Generated {len(new_results)} new results.")
    else:
        print("No new images found to process.")

    # --- Combine and Save Results ---
    if new_results:
        print("Combining new results with existing data...")
        final_results = existing_results + new_results
        print(f"Total results to save: {len(final_results)}")
        print(f"Saving combined results to {OUTPUT_FILE}...")
        try:
            # Write to a temporary file first, then rename for atomicity
            temp_output_file = OUTPUT_FILE + ".tmp"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=4, ensure_ascii=False)
            os.replace(temp_output_file, OUTPUT_FILE) # Atomic rename/replace
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving final results to JSON: {e}")
            if os.path.exists(temp_output_file):
                 try: os.remove(temp_output_file) # Clean up temp file on error
                 except OSError: pass
    elif existing_results:
         print("No new results generated. Existing results file remains unchanged.")
    else:
         print("No results generated and no existing results found.")


    print("\n===== Script Finished =====")
