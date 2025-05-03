# -*- coding: utf-8 -*-
import streamlit as st
import json
import pandas as pd # Keep pandas for potential display formatting if needed later
from PIL import Image
import os
import threading
import sqlite3 # Added for database interaction

# --- Configuration ---
# Path to the SQLite database created by the conversion script
DATABASE_PATH = "image_data.db"
VOTES_JSON_PATH = "caption_votes_data.json" # File to store persistent votes

# --- Project Links ---
GOOGLE_DRIVE_LINKS = {
    "Downloaded Bing Images": "https://drive.google.com/drive/folders/1ns0JRTqNpEp-v-RnKp6SOW4s3Y8q67ZT?usp=sharing",
    "User Collected Images": "https://drive.google.com/drive/folders/1OJlnBU7w9CTkimi-sMSS26vfoFOwellr?usp=sharing",
    "Scripts and Evals": "https://drive.google.com/drive/folders/1fo41IKZFZQr9WZMTV-veiYrU_B5TJBAO?usp=sharing"
}

# --- Lock for thread-safe file writing (for votes) ---
file_lock = threading.Lock()

# --- Database Helper Functions ---
@st.cache_resource # Cache the database connection for the session
def get_db_connection(db_path):
    """Creates and returns a connection to the SQLite database."""
    if not os.path.exists(db_path):
         st.error(f"Database file not found at {db_path}. Please run the JSON to SQLite conversion script first.")
         return None
    try:
        # check_same_thread=False is required for Streamlit as it uses threads
        conn = sqlite3.connect(db_path, check_same_thread=False)
        # Return rows as dictionary-like objects for easier access
        conn.row_factory = sqlite3.Row
        print(f"DB connection pool created for {db_path}")
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database '{db_path}': {e}")
        return None

# Use st.cache_data for functions that fetch data that changes infrequently
@st.cache_data(ttl=3600) # Cache distinct cultures for an hour
def fetch_distinct_values(_conn, column, table="images"):
    """Fetches distinct non-null values from a column. `_conn` is passed to enable caching based on connection."""
    if not _conn: return []
    try:
        # Use the actual connection object from the argument
        cur = _conn.cursor()
        cur.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL ORDER BY {column}")
        rows = cur.fetchall()
        return [row[column] for row in rows]
    except sqlite3.Error as e:
        st.error(f"Error fetching distinct {column} values: {e}")
        return []

@st.cache_data(ttl=3600) # Cache queries per culture for an hour
def fetch_queries_for_culture(_conn, culture, table="images"):
    """Fetches distinct non-null queries for a specific culture."""
    if not _conn: return []
    try:
        cur = _conn.cursor()
        cur.execute(f"""
            SELECT DISTINCT original_query
            FROM {table}
            WHERE culture = ? AND original_query IS NOT NULL
            ORDER BY original_query
        """, (culture,))
        rows = cur.fetchall()
        return [row['original_query'] for row in rows]
    except sqlite3.Error as e:
        st.error(f"Error fetching queries for culture '{culture}': {e}")
        return []

# Don't cache this usually, as it depends directly on filter selections
def fetch_image_paths_for_query(conn, culture, query, table="images"):
     """Fetches image paths for a specific culture and query."""
     if not conn: return []
     try:
        cur = conn.cursor()
        # Ensure parameters are passed correctly to prevent SQL injection vulnerabilities
        cur.execute(f"""
            SELECT image_path
            FROM {table}
            WHERE culture = ? AND original_query = ?
            ORDER BY image_path -- Or sort by filename if needed
        """, (culture, query))
        rows = cur.fetchall()
        return [row['image_path'] for row in rows]
     except sqlite3.Error as e:
        st.error(f"Error fetching image paths for query '{query}': {e}")
        return []

# Cache details for individual images as they are unlikely to change often within a session
@st.cache_data(ttl=3600)
def fetch_image_details(_conn, image_path, table="images"):
     """Fetches all details for a specific image path."""
     if not _conn: return None
     try:
        cur = _conn.cursor()
        cur.execute(f"SELECT * FROM {table} WHERE image_path = ?", (image_path,))
        row = cur.fetchone()
        # Convert sqlite3.Row to a plain dictionary for easier handling downstream
        return dict(row) if row else None
     except sqlite3.Error as e:
        st.error(f"Error fetching details for image '{image_path}': {e}")
        return None

# --- Vote Data Functions (remain the same, operating on JSON) ---
def load_votes_data(votes_path):
    """Loads the persistent total votes data from the votes JSON file."""
    if os.path.exists(votes_path):
        try:
            with file_lock:
                with open(votes_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                         valid_data = {}
                         for k, v in data.items():
                              # Ensure the loaded structure is correct
                              if isinstance(v, dict) and 'up' in v and 'down' in v:
                                   valid_data[k] = v
                              else:
                                   print(f"Warning: Invalid vote structure for key '{k}' in {votes_path}. Skipping.")
                         return valid_data
                    else:
                         # If the file content is not a dict, return empty
                         print(f"Warning: Votes file '{votes_path}' does not contain a dictionary. Starting fresh.")
                         return {}
        except json.JSONDecodeError:
            st.warning(f"Could not decode existing votes file '{votes_path}'. Starting fresh.")
            return {}
        except Exception as e:
            st.warning(f"Error loading votes file '{votes_path}': {e}. Starting fresh.")
            return {}
    else:
        return {} # Return empty dict if file doesn't exist

def save_votes_data(votes_path, votes_dict):
    """Saves the persistent total votes data dictionary to the votes JSON file."""
    temp_votes_path = votes_path + ".tmp" # Define temp path outside try block
    try:
        with file_lock: # Acquire lock for writing
            with open(temp_votes_path, 'w', encoding='utf-8') as f:
                json.dump(votes_dict, f, indent=4)
            os.replace(temp_votes_path, votes_path) # Atomic rename/replace
    except Exception as e:
        st.error(f"Error saving votes data to '{votes_path}': {e}")
        # Clean up temp file if it exists on error
        if os.path.exists(temp_votes_path):
            try: os.remove(temp_votes_path)
            except OSError: pass

def handle_vote(image_path, vote_type):
    """Handles vote click and saves updated votes to file."""
    # Ensure persistent votes dict exists for the image in session state
    if image_path not in st.session_state.votes:
        st.session_state.votes[image_path] = {'up': 0, 'down': 0}

    # Get the user's previous vote status for this image in this session
    previous_session_vote = st.session_state.user_session_vote_status.get(image_path, None)

    up_change, down_change, new_session_vote = 0, 0, None

    if vote_type == previous_session_vote: # Remove vote
        up_change = -1 if vote_type == 'up' else 0
        down_change = -1 if vote_type == 'down' else 0
        new_session_vote = None
    elif previous_session_vote is not None: # Change vote
        up_change = 1 if vote_type == 'up' else -1
        down_change = 1 if vote_type == 'down' else -1
        new_session_vote = vote_type
    else: # New vote
        up_change = 1 if vote_type == 'up' else 0
        down_change = 1 if vote_type == 'down' else 0
        new_session_vote = vote_type

    # Update persistent total counts in session state first
    st.session_state.votes[image_path]['up'] = max(0, st.session_state.votes[image_path]['up'] + up_change)
    st.session_state.votes[image_path]['down'] = max(0, st.session_state.votes[image_path]['down'] + down_change)

    # Update the user's vote status for this session
    st.session_state.user_session_vote_status[image_path] = new_session_vote

    # Persist the updated *total* counts to the JSON file
    save_votes_data(VOTES_JSON_PATH, st.session_state.votes)

# --- Display Function ---
def display_entry(conn, image_path): # Takes connection and image_path now
    """Fetches details for an image path from DB and displays them."""
    # Fetch details using the cached function
    entry_data = fetch_image_details(conn, image_path)
    if not entry_data:
        st.warning(f"Could not retrieve details for image: {image_path}")
        return

    # Extract data from the dictionary
    caption = entry_data.get('caption', 'N/A')
    translated_caption = entry_data.get('translated_caption', None)
    # Load Q&A pairs from JSON string stored in DB
    try:
        qa_pairs_str = entry_data.get('qa_pairs', '[]') # Default to empty list string
        qa_pairs = json.loads(qa_pairs_str) if qa_pairs_str else []
        if not isinstance(qa_pairs, list): # Ensure it's a list after loading
             qa_pairs = []
    except json.JSONDecodeError:
        qa_pairs = []
        st.warning(f"Could not parse Q&A data for {os.path.basename(image_path)}")

    original_query = entry_data.get('original_query', 'N/A')
    culture = entry_data.get('culture', 'N/A')

    # Get votes (still managed via session state and separate file)
    total_votes = st.session_state.votes.get(image_path, {'up': 0, 'down': 0})
    session_vote = st.session_state.user_session_vote_status.get(image_path, None)

    st.markdown("---") # Separator
    col1, col2 = st.columns([1, 2]) # Image column, Text column

    with col1:
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                st.image(img, use_container_width=True)
            except Exception as img_e:
                st.warning(f"Could not load image: {os.path.basename(image_path)}\nError: {img_e}")
        else:
            st.warning(f"Image file not found at: {image_path}")

    with col2:
        st.markdown(f"**Culture:** {culture}")
        st.markdown(f"**Original Query:** `{original_query}`")
        st.markdown(f"**File:** `{os.path.basename(image_path)}`")

        # Display Captions
        st.markdown("**Caption (English):**")
        st.markdown(f"> {caption if caption else 'N/A'}")

        # Voting Buttons
        if caption and image_path != 'N/A':
             vote_cols = st.columns([1, 1, 5])
             up_button_type = "primary" if session_vote == 'up' else "secondary"
             down_button_type = "primary" if session_vote == 'down' else "secondary"
             with vote_cols[0]:
                  st.button(f"👍 ({total_votes['up']})", key=f"up_{image_path}", on_click=handle_vote, args=(image_path, 'up'), type=up_button_type)
             with vote_cols[1]:
                  st.button(f"👎 ({total_votes['down']})", key=f"down_{image_path}", on_click=handle_vote, args=(image_path, 'down'), type=down_button_type)

        # Display Translated Caption
        if translated_caption:
            st.markdown(f"**Caption ({culture}):**")
            st.markdown(f"> {translated_caption}")

        # Display Q&A Pairs
        if qa_pairs and isinstance(qa_pairs, list):
             with st.expander("Show Q&A Pairs", expanded=False):
                 for i, qa in enumerate(qa_pairs):
                      if not isinstance(qa, dict): continue
                      st.markdown(f"**{i+1}. Type: {qa.get('type', 'N/A')}**")
                      st.markdown(f"   - **Q (Eng):** {qa.get('question', 'N/A')}")
                      st.markdown(f"   - **A (Eng):** {qa.get('answer', 'N/A')}")
                      if qa.get('options'):
                           st.markdown(f"   - **Options:**")
                           for opt in qa['options']: st.markdown(f"     - {opt}")
                      if qa.get('translated_question'): st.markdown(f"   - **Q ({culture}):** {qa['translated_question']}")
                      if qa.get('translated_answer'): st.markdown(f"   - **A ({culture}):** {qa['translated_answer']}")
                      st.markdown("---")
        elif not qa_pairs:
             st.markdown("_No Q&A pairs generated for this image._")


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("🌍 Afri-Aya Dataset Viewer (SQLite Backend)")

# --- Initialize/Load Session State & DB Connection ---
# Load persistent total votes ONCE per session start
if 'votes' not in st.session_state:
    print(f"Loading persistent votes from {VOTES_JSON_PATH} into session state...")
    st.session_state.votes = load_votes_data(VOTES_JSON_PATH)
    print(f"Loaded {len(st.session_state.votes)} total vote records.")
# Initialize session-specific vote tracking ONCE per session start
if 'user_session_vote_status' not in st.session_state:
     print("Initializing user session vote status...")
     st.session_state.user_session_vote_status = {}

# Get Database Connection (cached)
conn = get_db_connection(DATABASE_PATH)

# --- Sidebar ---
st.sidebar.header("Filters")

# Display filters only if DB connection is successful
if conn:
    # Pass connection to cached functions
    cultures = fetch_distinct_values(conn, "culture")
    if not cultures:
        st.sidebar.warning("No cultures found in the database.")
        selected_culture = None
    else:
        selected_culture = st.sidebar.selectbox("Select Culture/Language", cultures)

    if selected_culture:
        # Pass connection to cached functions
        queries = fetch_queries_for_culture(conn, selected_culture)
        if not queries:
            st.sidebar.warning(f"No original queries found for culture '{selected_culture}'.")
            selected_query = None
        else:
            selected_query = st.sidebar.selectbox("Select Original Query", queries)
            st.sidebar.info(f"Filters selected:\n- Culture: {selected_culture}\n- Query: '{selected_query}'")
    else:
        selected_query = None
else:
    st.sidebar.error("Database connection failed. Cannot display filters.")
    selected_culture = None
    selected_query = None

# --- Add Project Resource Links to Sidebar ---
st.sidebar.markdown("---") # Separator
st.sidebar.header("Project Resources")
for name, url in GOOGLE_DRIVE_LINKS.items():
    st.sidebar.markdown(f"- [{name}]({url})")
st.sidebar.markdown("---") # Separator

# --- Main Display Area ---
if conn and selected_culture and selected_query:
    # Fetch only the image paths matching the filters
    image_paths_to_display = fetch_image_paths_for_query(conn, selected_culture, selected_query)

    if image_paths_to_display:
        st.header(f"Images for '{selected_query}' ({selected_culture}) - {len(image_paths_to_display)} found")
        # Display details for each image path by querying the DB individually
        for img_path in image_paths_to_display:
            display_entry(conn, img_path) # Pass connection and path
    else:
        st.warning("No images found for the selected culture and query combination in the database.")

elif conn:
     # Handles cases where filters couldn't be populated or selections not made
     st.info("Select a culture and query from the sidebar to view images.")
else:
     # Error message already shown if conn is None
     st.info("Cannot display images due to database connection failure.")


# Note: The database connection is kept open for the duration of the session
# due to @st.cache_resource. Streamlit handles closing it appropriately.
