import cv2
import pytesseract
import sqlite3
import os
import numpy as np
from datetime import datetime
import streamlit as st
from PIL import Image

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\VIKRANT\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Constants
DB_FOLDER = "vehicle_data"
DB_PATH = os.path.join(DB_FOLDER, "vehicle_data.db")
PREDEFINED_VEHICLES = [
    {"plate_number": "RJ14CV0002", "entry_time": "2024-11-27 09:00:00", "last_entry_time": "2024-11-27 09:00:00"},
    {"plate_number": "22BH6517A", "entry_time": "2024-11-27 09:10:00", "last_entry_time": "2024-11-27 09:10:00"},
    {"plate_number": "KA18EQ0001", "entry_time": "2024-11-27 09:20:00", "last_entry_time": "2024-11-27 09:20:00"},
    {"plate_number": "KL65AN7722", "entry_time": "2024-11-27 09:30:00", "last_entry_time": "2024-11-27 09:30:00"},
]

# Create database folder
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# Initialize SQLite database
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        plate_number TEXT UNIQUE,
                        entry_time TEXT,
                        last_entry_time TEXT)''')
    for vehicle in PREDEFINED_VEHICLES:
        try:
            cursor.execute(
                "INSERT INTO logs (plate_number, entry_time, last_entry_time) VALUES (?, ?, ?)",
                (vehicle["plate_number"], vehicle["entry_time"], vehicle["last_entry_time"])
            )
        except sqlite3.IntegrityError:
            pass  # Ignore duplicates
    conn.commit()
    conn.close()

# Log license plate
def log_plate(plate_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plate_number = plate_number.replace(" ", "").upper()

    cursor.execute("SELECT * FROM logs WHERE plate_number = ?", (plate_number,))
    existing_plate = cursor.fetchone()

    if existing_plate is None:
        cursor.execute("INSERT INTO logs (plate_number, entry_time, last_entry_time) VALUES (?, ?, ?)",
                       (plate_number, timestamp, timestamp))
        conn.commit()
        conn.close()
        return True, None
    else:
        last_entry_time = existing_plate[3]
        cursor.execute("UPDATE logs SET last_entry_time = ? WHERE plate_number = ?",
                       (timestamp, plate_number))
        conn.commit()
        conn.close()
        return False, last_entry_time

# Process image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    return edged

# Find license plate contour
def find_license_plate_contour(edged):
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if 1.5 < aspect_ratio < 5 and area > 1000:
            return contour
    return None

# Extract license plate
def extract_license_plate(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w]

# Preprocess for OCR
def preprocess_for_ocr(license_plate):
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_plate

# Perform OCR
def perform_ocr(license_plate):
    processed_plate = preprocess_for_ocr(license_plate)
    text = pytesseract.image_to_string(processed_plate, config="--psm 8")
    text = text.replace('O', '0').replace('I', '1').replace('l', '1').replace('Z', '2')
    text = text.replace('S', '5').replace('B', '8').replace(' ', '')
    return text.strip()

# Display database entries
def view_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM logs")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Streamlit app
def main():
    st.title("Vehicle Monitoring System")
    st.sidebar.title("Options")
    mode = st.sidebar.selectbox("Select Mode", ["Home", "Run Detection", "View Database"])

    if mode == "Home":
        st.write("Welcome to the Vehicle Monitoring System!")
        st.write("Use the sidebar to navigate between modes.")

    elif mode == "Run Detection":
        st.write("### Vehicle Detection")
        run_detection = st.button("Start Camera")

        if run_detection:
            # Capture video feed
            st.write("Capturing video... Press 'q' to quit the capture.")
            video_capture = cv2.VideoCapture(0)
            FRAME_WINDOW = st.image([])

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                edged = preprocess_image(frame)
                contour = find_license_plate_contour(edged)

                if contour is not None:
                    license_plate = extract_license_plate(frame, contour)
                    plate_text = perform_ocr(license_plate)

                    if plate_text:
                        is_new_car, last_entry_time = log_plate(plate_text)
                        label = "New Car" if is_new_car else f"Existing Car - Last Entry: {last_entry_time}"
                        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Update Streamlit image
                FRAME_WINDOW.image(frame, channels="BGR")

                # Quit capture with 'q'
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            video_capture.release()
            cv2.destroyAllWindows()

    elif mode == "View Database":
        st.write("### Database Entries")
        rows = view_database()
        st.table(rows)

if __name__ == "__main__":
    init_database()
    main()
