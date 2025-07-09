import cv2
import numpy as np
from datetime import datetime
import os
import re
from config import PLATE, APP # Assuming config.py exists and works

# Import YOLO and EasyOCR
from ultralytics import YOLO
import easyocr
import math # For perspective correction calculations

# Access values from config
PLATE_MIN_AREA = PLATE.get('MIN_AREA', 800) # This might become less relevant with YOLO, but good to keep.
PLATE_MAX_RATIO = PLATE.get('MAX_RATIO', 6.0) # Might also be less relevant
DEBUG_MODE = APP.get('DEBUG_MODE', False)
SAVE_IMAGES = APP.get('SAVE_IMAGES', False)

class PlateRecognizer:
    def __init__(self, gsheet_handler_instance, debug_mode=DEBUG_MODE, save_images=SAVE_IMAGES):
        self.gsheet_handler = gsheet_handler_instance
        self.DEBUG_MODE = debug_mode
        self.SAVE_IMAGES = save_images

        # ----------------------------------------------------
        # Initialize YOLOv8 Model for Plate Detection
        # IMPORTANT: Replace 'yolov8n.pt' with your trained license plate model if you have one.
        # If you are using a general model like 'yolov8n.pt', you'll need to know the
        # class ID for 'license plate' if it was part of its training.
        # For simplicity, we'll assume it's directly detecting license plates.
        self.yolo_model = YOLO('C:/Users/sarth/OneDrive/Desktop/Final_ALPR/runs/detect/train/weights/best.pt')
        
        # Load EasyOCR
        # 'en' for English, add 'hi' for Hindi if relevant for state names
        # 'gpu=True' for GPU acceleration if available, otherwise 'gpu=False'
        self.easy_ocr_reader = easyocr.Reader(['en'], gpu=True) # Set gpu=False if no NVIDIA GPU
        print("✅ YOLOv8 Model and EasyOCR Reader initialized.")
        # ----------------------------------------------------

        self.INDIAN_STATE_CODES = [
            'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 
            'JK', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PB', 
            'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB',
            # Newly added / recently recognized UT codes if applicable:
            'AN', 'DN', 'DD', 'LA' # Andaman & Nicobar, Dadra & Nagar Haveli, Daman & Diu, Ladakh
        ]
        # -----------------------------------------------------------

        # --- REVISED CHAR_REPLACEMENTS: ONLY UNAMBIGUOUS CASES ---
        self.CHAR_REPLACEMENTS = {
            'O': '0', 'Q': '0', 'D': '0', # D sometimes misread as 0
            'I': '1', 'L': '1',
            'S': '5',
            'G': '6',
            'B': '8',
            'Z': '2',
            # Add any other consistently misread character pairs where one is clearly intended
            # e.g., if 'A' is *always* misread as '4' in a digit position, add 'A':'4'
        }
        # ---------------------------------------------------------

        self.STANDARD_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$")
        self.STANDARD_PLATE_REGEX_V2 = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1}[0-9]{4}$")
        self.STANDARD_PLATE_REGEX_NO_SERIES = re.compile(r"^[A-Z]{2}[0-9]{1,2}[0-9]{4}$")
        self.BH_PLATE_REGEX = re.compile(r"^[0-9]{2}BH[0-9]{2}[A-Z]{2}[0-9]{4}$")

        self.ocr_window_created = False

    def order_points(self, pts):
        # Initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # The sum of the (x, y) coordinates will be the smallest for the top-left and
        # largest for the bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        # The difference between the points will be the smallest for the top-right
        # and largest for the bottom-left
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left

        return rect

    def four_point_transform(self, image, pts):
        # Obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute the width of the new image, which will be the
        # maximum distance between the bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "bird's eye view",
        # (i.e. top-down view) of the image, specifying points in the
        # top-left, top-right, bottom-right, and bottom-left order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # Return the warped image
        return warped


    def clean_plate_text(self, text):
        """
        Cleans and corrects OCR output with improved handling for 'IND' and
        more targeted positional corrections.
        """
        if not text:
            return ""

        # Step 1: Remove "IND" if present at the beginning (case-insensitive)
        text = re.sub(r'^\s*IND\s*', '', text, flags=re.IGNORECASE)

        # Step 2: Initial cleanup - remove non-alphanumeric (except space initially), newlines, and strip spaces
        # Convert to uppercase immediately
        initial_cleaned_text = re.sub(r'[^A-Z0-9\s]', '', text.upper()).strip() # Keep space to handle it later
        initial_cleaned_text = initial_cleaned_text.replace(" ", "") # Remove all spaces after initial cleanup

        if not initial_cleaned_text:
            return ""

        # Step 3: Apply broad character replacements
        processed_text_list = list(initial_cleaned_text)
        for i, char in enumerate(processed_text_list):
            processed_text_list[i] = self.CHAR_REPLACEMENTS.get(char, char)
        processed_text = "".join(processed_text_list)
        
        # DEBUG_MODE checks moved to recognize_plate
        # if self.DEBUG_MODE:
        # print(f"DEBUG: After CHAR_REPLACEMENTS: '{processed_text}'")

        # Step 4: Apply positional heuristics more carefully
        final_chars = list(processed_text)
        
        # Ensure first two chars are letters (State Code)
        if len(final_chars) >= 2:
            # If the first char is a digit, try to convert it to a likely letter
            if final_chars[0].isdigit():
                final_chars[0] = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', '9':'P'}.get(final_chars[0], 'A') # Fallback to 'A' if no good mapping
            # Same for the second char
            if final_chars[1].isdigit():
                final_chars[1] = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', '9':'P'}.get(final_chars[1], 'B') # Fallback to 'B'
            
            # Specific M/H handling: If first two chars are 'HH' and MH is a valid state, try to correct
            if "".join(final_chars[:2]) == 'HH' and 'MH' in self.INDIAN_STATE_CODES:
                final_chars[0] = 'M' # Assuming M is more likely if the detection is clean enough for HH
            elif "".join(final_chars[:2]) == 'HM' and 'MH' in self.INDIAN_STATE_CODES:
                final_chars[0] = 'M' # Correct H to M
            elif "".join(final_chars[:2]) == 'IM' and 'MH' in self.INDIAN_STATE_CODES: # If I was misread as 1 and then became I
                final_chars[0] = 'M' # Correct I to M (very specific heuristic)
            
            # Check for HR specific cases
            if "".join(final_chars[:2]) == 'HR' and final_chars[2].isalpha() and len(final_chars) > 3 and final_chars[3].isalpha():
                # If HR is followed by two letters where two digits are expected (e.g., HRZC for HR26)
                if final_chars[2] in self.CHAR_REPLACEMENTS.keys() and final_chars[3] in self.CHAR_REPLACEMENTS.keys():
                    final_chars[2] = self.CHAR_REPLACEMENTS.get(final_chars[2], final_chars[2])
                    final_chars[3] = self.CHAR_REPLACEMENTS.get(final_chars[3], final_chars[3])

        # Ensure numeric parts are indeed numbers
        # This loop now iterates over potential digit positions based on common Indian plate format
        # State Code (2 chars), District Code (2 digits), Series (1-2 chars), Number (4 digits)
        # So, digits are expected at positions 2, 3, and from length-4 to length-1
        
        # District Code (positions 2 and 3)
        if len(final_chars) > 2 and final_chars[2].isalpha():
            final_chars[2] = {'O':'0', 'I':'1', 'S':'5', 'G':'6', 'B':'8', 'Z':'2', 'A':'4'}.get(final_chars[2], final_chars[2])
        if len(final_chars) > 3 and final_chars[3].isalpha():
            final_chars[3] = {'O':'0', 'I':'1', 'S':'5', 'G':'6', 'B':'8', 'Z':'2', 'A':'4'}.get(final_chars[3], final_chars[3])
        
        # Last 4 characters (number part)
        if len(final_chars) >= 4:
            for i in range(max(0, len(final_chars) - 4), len(final_chars)):
                if final_chars[i].isalpha():
                    final_chars[i] = {'O':'0', 'I':'1', 'S':'5', 'G':'6', 'B':'8', 'Z':'2', 'A':'4'}.get(final_chars[i], final_chars[i])


        final_cleaned_text = "".join(final_chars)
        
        # DEBUG_MODE checks moved to recognize_plate
        # if self.DEBUG_MODE:
        # print(f"DEBUG: After Positional Heuristics: '{final_cleaned_text}'")

        return final_cleaned_text

    def validate_indian_plate(self, plate_text):
        """
        Validates if the cleaned plate text matches Indian license plate patterns.
        Adds more debug output.
        """
        if not plate_text:
            # DEBUG_MODE check moved to recognize_plate
            # if self.DEBUG_MODE:
            # print("DEBUG: Validation: Plate text is empty.")
            return False

        # Ensure state code is valid (e.g., MH, RJ, DL)
        if len(plate_text) >= 2:
            state_code = plate_text[:2]
            if state_code not in self.INDIAN_STATE_CODES:
                # DEBUG_MODE check moved to recognize_plate
                # if self.DEBUG_MODE:
                # print(f"DEBUG: Validation: Invalid State Code '{state_code}'.")
                return False
        else:
            # DEBUG_MODE check moved to recognize_plate
            # if self.DEBUG_MODE:
            # print("DEBUG: Validation: Plate text too short for state code check.")
            return False

        # Test against different regex patterns in a logical order
        if self.STANDARD_PLATE_REGEX.fullmatch(plate_text):
            # DEBUG_MODE check moved to recognize_plate
            # if self.DEBUG_MODE:
            # print(f"DEBUG: Validation: Matched STANDARD_PLATE_REGEX for '{plate_text}'.")
            return True
        elif self.STANDARD_PLATE_REGEX_V2.fullmatch(plate_text):
            # DEBUG_MODE check moved to recognize_plate
            # if self.DEBUG_MODE:
            # print(f"DEBUG: Validation: Matched STANDARD_PLATE_REGEX_V2 for '{plate_text}'.")
            return True
        elif self.STANDARD_PLATE_REGEX_NO_SERIES.fullmatch(plate_text):
            # DEBUG_MODE check moved to recognize_plate
            # if self.DEBUG_MODE:
            # print(f"DEBUG: Validation: Matched STANDARD_PLATE_REGEX_NO_SERIES for '{plate_text}'.")
            return True
        elif self.BH_PLATE_REGEX.fullmatch(plate_text):
            # DEBUG_MODE check moved to recognize_plate
            # if self.DEBUG_MODE:
            # print(f"DEBUG: Validation: Matched BH_PLATE_REGEX for '{plate_text}'.")
            return True
        
        # DEBUG_MODE check moved to recognize_plate
        # if self.DEBUG_MODE:
        # print(f"DEBUG: Validation: No regex match for '{plate_text}'.")
        return False

    def recognize_plate(self, img):
        detected_plate_text = None
        plate_roi = None
        bbox_coords = None # To store [x1, y1, x2, y2] of the detected plate
        
        try:
            # ----------------------------------------------------
            # Step 1: Detect plates using YOLOv8
            results = self.yolo_model(img, verbose=False)[0] # Run inference, get the first result object

            # Filter for license plate detections (assuming class 2 is 'license plate' for a general model,
            # or if you fine-tuned a model just for plates, it would be class 0 usually).
            # You might need to adjust this based on your YOLO model's classes.
            # Example: license_plate_class_id = 2 # Check your model's class list
            
            # Find the best (largest or highest confidence) license plate detection
            best_conf = 0
            best_bbox = None
            
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                
                # If your YOLO model is specifically trained *only* for license plates,
                # then all detections are likely plates. Just check confidence.
                if score > 0.5: # Adjust confidence threshold as needed
                    current_w = x2 - x1
                    current_h = y2 - y1
                    current_area = current_w * current_h
                    # Prioritize larger plates for better OCR
                    if best_bbox is None or current_area > (best_bbox[2]-best_bbox[0])*(best_bbox[3]-best_bbox[1]):
                        best_conf = score
                        best_bbox = [int(x1), int(y1), int(x2), int(y2)]

            if best_bbox:
                x1, y1, x2, y2 = best_bbox
                plate_roi = img[y1:y2, x1:x2]
                bbox_coords = (x1, y1, x2-x1, y2-y1) # For debug display in old format

                # ----------------------------------------------------
                # Step 2: (Optional but Recommended) Perspective Correction on plate_roi
                # For now, we'll just use the cropped `plate_roi` directly for EasyOCR.

                # ----------------------------------------------------
                # Step 3: Recognize text using EasyOCR
                results_ocr = self.easy_ocr_reader.readtext(plate_roi)

                raw_ocr_text_candidates = []
                for (bbox_ocr, text_ocr, prob_ocr) in results_ocr:
                    raw_ocr_text_candidates.append(text_ocr)
                    if self.DEBUG_MODE: # Only draw OCR boxes if DEBUG_MODE is active
                        p0, p1, p2, p3 = bbox_ocr
                        cv2.rectangle(plate_roi, (int(p0[0]), int(p0[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 1) # Yellow boxes
                        cv2.putText(plate_roi, text_ocr, (int(p0[0]), int(p0[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Combine all recognized text lines into one string, separated by space or nothing
                raw_combined_text = " ".join(raw_ocr_text_candidates)

                if self.DEBUG_MODE:
                    print(f"DEBUG: EasyOCR Raw Text Candidates: {raw_ocr_text_candidates}", flush=True)
                    print(f"DEBUG: EasyOCR Combined Raw Text: '{raw_combined_text}'", flush=True)

                # Step 4: Clean and Validate EasyOCR Output
                cleaned_attempt_text = self.clean_plate_text(raw_combined_text)
                temp_validated_text = self.validate_indian_plate(cleaned_attempt_text)

                if self.DEBUG_MODE:
                    print(f"DEBUG: After CHAR_REPLACEMENTS: '{cleaned_attempt_text}'") # This was moved here
                    print(f"DEBUG: After Positional Heuristics: '{cleaned_attempt_text}'") # This was moved here
                    # Specific validation debug prints
                    if not cleaned_attempt_text:
                        print("DEBUG: Validation: Plate text is empty after cleaning.")
                    elif len(cleaned_attempt_text) >= 2 and cleaned_attempt_text[:2] not in self.INDIAN_STATE_CODES:
                        print(f"DEBUG: Validation: Invalid State Code '{cleaned_attempt_text[:2]}' for '{cleaned_attempt_text}'.")
                    elif not temp_validated_text:
                        print(f"DEBUG: Validation: No regex match for '{cleaned_attempt_text}'.")

                    print(f"DEBUG: Cleaned Attempt Text: '{cleaned_attempt_text}'", flush=True)
                    print(f"DEBUG: Validation Result: '{temp_validated_text}'", flush=True)


                if temp_validated_text:
                    detected_plate_text = cleaned_attempt_text
                    if self.DEBUG_MODE:
                        print(f"DEBUG: Valid plate recognized: '{detected_plate_text}'", flush=True)

                # --- Debug output (show result of final attempt, or the validated one) ---
                # This part now only executes if a plate was detected by YOLO (best_bbox is True)
                if self.DEBUG_MODE:
                    display_text_for_debug = detected_plate_text if detected_plate_text else (raw_combined_text if plate_roi is not None else "No Plate Detected in ROI")
                    self._show_debug_output(img, display_text_for_debug, bbox_coords, plate_roi)
                    print(f"DEBUG: Final Detected Plate Text (to be logged): '{detected_plate_text}'", flush=True)

                # Save image if a valid plate is found
                if detected_plate_text and self.SAVE_IMAGES:
                    self._save_plate_image(img, detected_plate_text)

                # Log to Google Sheets ONLY if a valid plate was detected
                if detected_plate_text:
                    print(f"Plate Detected: {detected_plate_text}", flush=True) # Non-debug output for successful detection
                    if self.gsheet_handler:
                        self.gsheet_handler.log_to_sheet(detected_plate_text)
                    else:
                        print("WARNING: Google Sheets handler not initialized. Skipping logging.", flush=True)
                else:
                    # If best_bbox was found but not a valid plate recognized, still consider it a "detection" for debug output
                    if self.DEBUG_MODE:
                        print(f"DEBUG: Plate detected by YOLO but no valid text recognized for logging.", flush=True)
            # If no best_bbox (no plate detected by YOLO), no output except initial "YOLO model initialized"
            else:
                pass # No plate detected, so no output

        except Exception as e:
            if self.DEBUG_MODE:
                print(f"[ERROR] Plate recognition failed: {str(e)}", flush=True)
                # Ensure the debug output still shows something on error if DEBUG_MODE is active
                self._show_debug_output(img, "ERROR", bbox_coords, plate_roi)

        return detected_plate_text

    def _show_debug_output(self, img, text, bbox=None, plate_img=None):
        debug_img = img.copy()

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 2)
            status_prefix = "Valid: " if self.validate_indian_plate(text) else "Invalid: "
            cv2.putText(debug_img, status_prefix + text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0) if status_prefix.startswith('Valid') else (0,0,255), 2)

        cv2.imshow('Detection', debug_img)

        if plate_img is not None and plate_img.size > 0: # Check if plate_img is not empty
            # Resize plate_img for consistent display
            display_plate_img = cv2.resize(plate_img, (300, 100), interpolation=cv2.INTER_AREA)
            cv2.imshow('OCR Input (Cropped Plate)', display_plate_img)
            self.ocr_window_created = True
        elif self.ocr_window_created:
            cv2.destroyWindow('OCR Input (Cropped Plate)')
            self.ocr_window_created = False

        cv2.waitKey(1)

    def _save_plate_image(self, img, plate_text):
        os.makedirs("captures", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_plate_text = ''.join(filter(str.isalnum, plate_text))
        filename = f"captures/{sanitized_plate_text}_{timestamp}.jpg"
        cv2.imwrite(filename, img)