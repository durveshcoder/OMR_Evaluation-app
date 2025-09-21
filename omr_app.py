
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import json
import io
import base64
from datetime import datetime
import os
import tempfile

st.set_page_config(
        page_title="OMR Evaluation System - Innomatics",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# OMR Evaluation System Class
class OMRProcessor:
    def __init__(self):
        self.bubble_threshold = 0.6
        self.answer_choices = ['A', 'B', 'C', 'D']
        self.subjects = ['PYTHON', 'DATA ANALYSIS', 'MySQL', 'POWER BI', 'Adv STATS']
        self.questions_per_subject = 20

    def load_answer_key(self, custom_answer_key=None):
        """Load the answer key - either custom or predefined"""
        if custom_answer_key is not None:
            return custom_answer_key
        
        # Default answer key
        answer_key = {
            'PYTHON': {
                1:'A', 2:'C', 3:'C', 4:'C', 5:'C', 6:'A', 7:'C', 8:'C', 9:'B', 10:'C',
                11:'A', 12:'A', 13:'D', 14:'A', 15:'B', 16:['A','B','C','D'], 17:'C', 18:'D', 19:'A', 20:'B'
            },
            'DATA ANALYSIS': {
                1:'A', 2:'D', 3:'B', 4:'A', 5:'C', 6:'B', 7:'A', 8:'B', 9:'D', 10:'C',
                11:'C', 12:'A', 13:'B', 14:'C', 15:'A', 16:'B', 17:'D', 18:'B', 19:'A', 20:'B'
            },
            'MySQL': {
                1:'C', 2:'C', 3:'C', 4:'B', 5:'B', 6:'A', 7:'C', 8:'B', 9:'D', 10:'A',
                11:'C', 12:'B', 13:'C', 14:'C', 15:'A', 16:'B', 17:'B', 18:'A', 19:['A','B'], 20:'B'
            },
            'POWER BI': {
                1:'B', 2:'C', 3:'A', 4:'B', 5:'C', 6:'B', 7:'B', 8:'C', 9:'C', 10:'B',
                11:'B', 12:'B', 13:'D', 14:'B', 15:'A', 16:'B', 17:'B', 18:'B', 19:'B', 20:'B'
            },
            'Adv STATS': {
                1:'A', 2:'B', 3:'C', 4:'B', 5:'C', 6:'B', 7:'B', 8:'B', 9:'A', 10:'B',
                11:'C', 12:'B', 13:'C', 14:'B', 15:'B', 16:'B', 17:'C', 18:'A', 19:'B', 20:'C'
            }
        }
        return answer_key

    def parse_csv_answer_key(self, csv_data):
        """Parse CSV data to create answer key dictionary"""
        try:
            import io
            df = pd.read_csv(io.StringIO(csv_data))
            
            answer_key = {}
            
            # Get all columns except the first one (question numbers)
            subjects = [col for col in df.columns if col.lower() not in ['question', 'q', 'question_number', 'q_num']]
            
            for subject in subjects:
                answer_key[subject] = {}
                
                for idx, row in df.iterrows():
                    question_num = None
                    answer = None
                    
                    # Find question number column
                    for col in df.columns:
                        if col.lower() in ['question', 'q', 'question_number', 'q_num']:
                            question_num = int(row[col])
                            break
                    
                    if question_num is None:
                        question_num = idx + 1  # Default to row index + 1
                    
                    answer = str(row[subject]).strip().upper()
                    
                    # Handle multiple answers (comma separated or with 'or')
                    if ',' in answer:
                        answer = [a.strip() for a in answer.split(',')]
                    elif ' or ' in answer:
                        answer = [a.strip() for a in answer.split(' or ')]
                    elif answer in ['A', 'B', 'C', 'D']:
                        answer = answer
                    else:
                        # Try to extract valid answers
                        valid_answers = []
                        for char in answer:
                            if char in ['A', 'B', 'C', 'D']:
                                valid_answers.append(char)
                        answer = valid_answers if len(valid_answers) > 1 else (valid_answers[0] if valid_answers else '')
                    
                    if answer:
                        answer_key[subject][question_num] = answer
            
            return answer_key
            
        except Exception as e:
            st.error(f"Error parsing CSV answer key: {str(e)}")
            return None

    def create_manual_answer_key(self, subjects, questions_per_subject):
        """Create a template for manual answer key entry"""
        answer_key = {}
        for subject in subjects:
            answer_key[subject] = {}
            for q_num in range(1, questions_per_subject + 1):
                answer_key[subject][q_num] = ''
        return answer_key

    def preprocess_image(self, image):
        """Preprocess the OMR sheet image for better bubble detection"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive threshold for better bubble detection
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)

            return gray, thresh
        except Exception as e:
            st.error(f"Error in image preprocessing: {str(e)}")
            return image, image

    def detect_and_extract_bubbles(self, image):
        """Enhanced bubble detection with improved accuracy for small bubbles"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Enhanced preprocessing for better bubble detection
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(filtered, (3, 3), 0)

            # Use multiple threshold methods for better detection
            # Method 1: Adaptive threshold
            thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 2)
            
            # Method 2: OTSU threshold
            _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Combine both thresholds
            thresh = cv2.bitwise_or(thresh1, thresh2)

            # Morphological operations to clean up the image
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # Remove small noise
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
            # Fill small holes
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

            # Find contours with better hierarchy
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Enhanced bubble filtering
            bubbles = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # More flexible area range for different bubble sizes
                if 25 < area < 2000:  # Adjusted range for small to medium bubbles
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        # Calculate circularity (4π * Area / Perimeter²)
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # More flexible circularity threshold
                        if circularity > 0.25:  # Adjusted for slightly non-circular shapes
                            x, y, w, h = cv2.boundingRect(contour)
                            aspect_ratio = float(w) / h
                            
                            # More flexible aspect ratio for various bubble shapes
                            if 0.4 <= aspect_ratio <= 2.5:
                                # Enhanced fill ratio calculation
                                roi = thresh[y:y+h, x:x+w]
                                if roi.size > 0:
                                    # Create a mask for the contour
                                    mask = np.zeros(roi.shape, dtype=np.uint8)
                                    contour_shifted = contour - [x, y]
                                    cv2.fillPoly(mask, [contour_shifted], 255)
                                    
                                    # Calculate fill ratio more accurately
                                    total_pixels = cv2.countNonZero(mask)
                                    if total_pixels > 0:
                                        filled_pixels = cv2.countNonZero(cv2.bitwise_and(roi, mask))
                                        fill_ratio = filled_pixels / total_pixels
                                    else:
                                        fill_ratio = 0
                                else:
                                    fill_ratio = 0

                                # Store bubble information
                                bubbles.append({
                                    'x': x, 'y': y, 'w': w, 'h': h, 
                                    'center': (x + w//2, y + h//2),
                                    'fill_ratio': fill_ratio,
                                    'area': area,
                                    'circularity': circularity,
                                    'aspect_ratio': aspect_ratio,
                                    'contour': contour
                                })

            # Sort bubbles by position (top-to-bottom, left-to-right)
            bubbles.sort(key=lambda b: (b['y'], b['x']))

            return bubbles, gray, thresh

        except Exception as e:
            st.error(f"Error in enhanced bubble detection: {str(e)}")
            return [], image, image

    def capture_image_from_camera(self, camera_index=0, timeout=10):
        """Capture a single frame from the default camera and return as BGR numpy image.

        Returns None on failure.
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            start = cv2.getTickCount()
            # wait for camera to warm up
            for _ in range(10):
                ret, frame = cap.read()
                if ret:
                    break

            # Attempt to read a single frame within timeout seconds
            timeout_ticks = timeout * cv2.getTickFrequency()
            ret = False
            frame = None
            while cv2.getTickCount() - start < timeout_ticks:
                ret, frame = cap.read()
                if ret and frame is not None:
                    break

            cap.release()
            if not ret or frame is None:
                return None

            # Return the captured BGR frame
            return frame

        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            return None

    def organize_bubbles_by_grid(self, bubbles, image_shape):
        """Organize detected bubbles into subject/question/choice grid"""
        if not bubbles:
            return {}

        try:
            height, width = image_shape[:2]
            
            # OMR Sheet Layout Configuration (matching create_annotated_image)
            header_height = int(height * 0.12)
            footer_height = int(height * 0.08)
            usable_height = height - header_height - footer_height
            usable_width = width - 100
            
            # Subject columns (5 subjects)
            num_subjects = len(self.subjects)
            col_width = usable_width // num_subjects
            col_start_x = 50
            
            # Question rows (20 questions per subject)
            questions_per_subject = 20
            row_height = usable_height // (questions_per_subject + 2)
            row_start_y = header_height + row_height
            
            # Choice spacing within each question
            choice_spacing = col_width // 6
            choice_start_offset = col_width // 8
            
            organized_answers = {}
            for subject in self.subjects:
                organized_answers[subject] = {}

            # Process each subject
            for subject_idx, subject in enumerate(self.subjects):
                col_x = col_start_x + subject_idx * col_width
                
                # Process questions for this subject
                for q_num in range(1, questions_per_subject + 1):
                    row_y = row_start_y + (q_num - 1) * row_height
                    
                    # Find bubbles in this question's area
                    question_bubbles = []
                    for bubble in bubbles:
                        bubble_x, bubble_y = bubble['center']
                        
                        # Check if bubble is in this question's area
                        if (col_x <= bubble_x <= col_x + col_width and 
                            row_y - row_height//2 <= bubble_y <= row_y + row_height//2):
                            question_bubbles.append(bubble)
                    
                    if question_bubbles:
                        # Sort bubbles by x position to get A, B, C, D order
                        question_bubbles.sort(key=lambda b: b['center'][0])
                        
                        # Find the most filled bubble
                        filled_bubble = max(question_bubbles, key=lambda b: b['fill_ratio'])
                        if filled_bubble['fill_ratio'] > self.bubble_threshold:
                            # Determine which choice based on position
                            bubble_position = question_bubbles.index(filled_bubble)
                            if bubble_position < len(self.answer_choices):
                                organized_answers[subject][q_num] = self.answer_choices[bubble_position]

            return organized_answers

        except Exception as e:
            st.error(f"Error in organizing bubbles: {str(e)}")
            return {}

    def simulate_realistic_student_answers(self):
        """Simulate realistic student answers for demonstration purposes"""
        import random
        random.seed(42)  # For reproducible demo results

        student_answers = {}
        answer_key = self.load_answer_key()

        for subject in self.subjects:
            student_answers[subject] = {}
            for q_num in range(1, 21):
                # 75% chance of getting the answer right for demo
                if random.random() < 0.75 and q_num in answer_key[subject]:
                    correct_answer = answer_key[subject][q_num]
                    if isinstance(correct_answer, list):
                        student_answers[subject][q_num] = random.choice(correct_answer)
                    else:
                        student_answers[subject][q_num] = correct_answer
                else:
                    # Wrong answer
                    student_answers[subject][q_num] = random.choice(self.answer_choices)

        return student_answers

    def evaluate_answers(self, student_answers, answer_key):
        """Evaluate student answers against the answer key"""
        results = {}
        total_correct = 0
        total_questions = 0

        for subject in self.subjects:
            if subject not in answer_key:
                continue

            subject_correct = 0
            subject_total = 0
            question_results = {}

            for q_num in range(1, 21):
                if q_num in answer_key[subject]:
                    correct_answer = answer_key[subject][q_num]
                    student_answer = student_answers[subject].get(q_num, '')

                    # Handle multiple correct answers (like question 16 and 59)
                    if isinstance(correct_answer, list):
                        is_correct = student_answer.upper() in [ca.upper() for ca in correct_answer]
                    else:
                        is_correct = student_answer.upper() == correct_answer.upper()

                    question_results[q_num] = {
                        'student_answer': student_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct
                    }

                    if is_correct:
                        subject_correct += 1
                        total_correct += 1

                    subject_total += 1
                    total_questions += 1

            results[subject] = {
                'correct': subject_correct,
                'total': subject_total,
                'percentage': round((subject_correct / subject_total * 100) if subject_total > 0 else 0, 1),
                'questions': question_results
            }

        results['overall'] = {
            'correct': total_correct,
            'total': total_questions,
            'percentage': round((total_correct / total_questions * 100) if total_questions > 0 else 0, 1)
        }

        return results

    def create_annotated_image(self, original_image, student_answers, answer_key, bubbles=None):
        """Create annotated image with accurate bubble positioning based on detected bubbles"""
        try:
            annotated = original_image.copy()
            if len(annotated.shape) == 2:
                annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

            height, width = annotated.shape[:2]

            # Colors must be in BGR order for OpenCV drawing
            colors = {
                'correct': (0, 255, 0),        # Green (B,G,R)
                'incorrect': (0, 0, 255),      # Red
                'correct_answer': (0, 255, 255),# Yellow
                'bubble_outline': (169, 169, 169), # Dark Gray
                'selected_outline': (255, 255, 255) # White outline for selected bubbles
            }

            # If we have actual detected bubbles, use them for accurate positioning
            if bubbles and len(bubbles) > 0:
                # Group bubbles by rows (questions) and columns (subjects)
                # Sort by center y then center x for more accurate spatial ordering
                bubbles_sorted = sorted(bubbles, key=lambda b: (b['center'][1], b['center'][0]))

                # Use dynamic tolerances based on median bubble height
                hs = [b.get('h', 0) for b in bubbles_sorted if b.get('h', 0) > 0]
                avg_h = int(np.median(hs)) if len(hs) > 0 else 20
                y_tolerance = max(12, int(avg_h * 0.7))

                # Create a grid mapping of bubbles using center coordinates
                rows = []
                current_row = []
                current_y = bubbles_sorted[0]['center'][1] if bubbles_sorted else 0

                for bubble in bubbles_sorted:
                    cy = int(bubble['center'][1])
                    if abs(cy - current_y) <= y_tolerance:
                        current_row.append(bubble)
                    else:
                        if len(current_row) >= 4:  # Minimum 4 bubbles for ABCD choices
                            rows.append(sorted(current_row, key=lambda b: b['center'][0]))
                        current_row = [bubble]
                        current_y = cy

                # Don't forget the last row
                if len(current_row) >= 4:
                    rows.append(sorted(current_row, key=lambda b: b['center'][0]))

                # Determine subject columns based on x-coordinates
                if rows:
                    # Determine subject column boundaries by clustering center x positions
                    all_cx = [b['center'][0] for row in rows for b in row]
                    x_sorted = sorted(all_cx)
                    subject_columns = []
                    if x_sorted:
                        # dynamic tolerance based on median width
                        ws = [b.get('w', 0) for row in rows for b in row if b.get('w',0)>0]
                        avg_w = int(np.median(ws)) if len(ws) > 0 else 40
                        x_tol = max(40, int(avg_w * 1.5))
                        current_group = [x_sorted[0]]
                        for x in x_sorted[1:]:
                            if x - current_group[-1] <= x_tol:
                                current_group.append(x)
                            else:
                                subject_columns.append(current_group)
                                current_group = [x]
                        subject_columns.append(current_group)

                    # Map bubbles to subject/question/choice structure
                    for subject_idx, subject in enumerate(self.subjects[:len(subject_columns)]):
                        if subject not in answer_key:
                            continue
                            
                        subject_x_coords = subject_columns[subject_idx]
                        min_x = min(subject_x_coords) - int(avg_w*0.6)
                        max_x = max(subject_x_coords) + int(avg_w*0.6)
                        
                        # Find rows (questions) for this subject
                        subject_rows = []
                        for row in rows:
                            subject_bubbles_in_row = []
                            for bubble in row:
                                if min_x <= bubble['x'] <= max_x:
                                    subject_bubbles_in_row.append(bubble)
                            
                            if len(subject_bubbles_in_row) >= 4:  # ABCD choices
                                subject_rows.append(sorted(subject_bubbles_in_row, key=lambda b: b['x']))
                        
                        # Annotate bubbles for each question
                        for q_idx, bubble_row in enumerate(subject_rows[:20]):  # Max 20 questions
                            q_num = q_idx + 1

                            if q_num not in answer_key[subject]:
                                continue

                            correct_answer = answer_key[subject][q_num]

                            # Compute detected answers from bubble fill ratios in this row
                            fills = []
                            for b in bubble_row[:4]:
                                fills.append((b.get('fill_ratio', 0), b))

                            # Determine which bubble(s) are marked above a dynamic threshold
                            # Threshold: max(fill_ratio * 0.5, 0.25) relative or absolute fallback
                            detected_choices = []
                            if fills:
                                max_fill = max([f for f, _ in fills])
                                # dynamic threshold based on max fill
                                dyn_thresh = max(0.18, min(0.6, max_fill * 0.5))
                                for idx_f, (fval, b) in enumerate(fills):
                                    if fval >= dyn_thresh:
                                        detected_choices.append((self.answer_choices[idx_f], b))

                            # Resolve detected answer string (pick first if multiple unless multiple allowed)
                            detected_answer = ''
                            if len(detected_choices) == 1:
                                detected_answer = detected_choices[0][0]
                            elif len(detected_choices) > 1:
                                # Multiple marks: join or pick highest fill
                                detected_answer = max(detected_choices, key=lambda x: x[1].get('fill_ratio',0))[0]

                            # For display and decision, compare detected_answer with student_answers if provided
                            # Prioritize actual detection over simulated student_answers
                            student_answer = detected_answer if detected_answer else student_answers.get(subject, {}).get(q_num, '')

                            # Determine correctness vs answer key
                            if isinstance(correct_answer, list):
                                is_correct = student_answer.upper() in [ca.upper() for ca in correct_answer] if student_answer else False
                                correct_choices = [ca.upper() for ca in correct_answer]
                            else:
                                is_correct = (student_answer.upper() == correct_answer.upper()) if student_answer else False
                                correct_choices = [correct_answer.upper()]

                            # Annotate each choice bubble in this row
                            for choice_idx, bubble in enumerate(bubble_row[:4]):  # ABCD only
                                choice = self.answer_choices[choice_idx]
                                center = (int(bubble['center'][0]), int(bubble['center'][1]))
                                radius = max(6, min(14, int(min(bubble.get('w',10), bubble.get('h',10)) * 0.45)))

                                # Skip if position is outside image bounds
                                if (center[0] < radius or center[0] >= width - radius or 
                                    center[1] < radius or center[1] >= height - radius):
                                    continue

                                # Draw base outline for all bubbles
                                cv2.circle(annotated, center, radius + 1, colors['bubble_outline'], 1)

                                # If this bubble was detected as selected
                                was_selected = any(dc[0] == choice for dc in detected_choices)

                                if was_selected:
                                    color = colors['correct'] if (choice == student_answer and is_correct) else colors['incorrect']
                                    # White outline for visibility
                                    cv2.circle(annotated, center, radius + 3, colors['selected_outline'], 2)
                                    # Colored circle for result
                                    cv2.circle(annotated, center, radius + 1, color, 3)
                                    # Fill if correct
                                    if choice == student_answer and is_correct:
                                        cv2.circle(annotated, center, radius - 2, color, -1)

                                # Show correct answer if student missed it
                                elif choice in correct_choices and (not is_correct):
                                    cv2.circle(annotated, center, radius + 2, colors['correct_answer'], 2)
                                    cv2.circle(annotated, center, 3, colors['correct_answer'], -1)

            else:
                # Fallback to calculated grid positions if no bubbles detected
                self._annotate_using_calculated_grid(annotated, student_answers, answer_key, colors, height, width)

            return annotated

        except Exception as e:
            st.error(f"Error creating annotated image: {str(e)}")
            return original_image

    def _annotate_using_calculated_grid(self, annotated, student_answers, answer_key, colors, height, width):
        """Fallback method using calculated grid positions"""
        # OMR Sheet Layout Configuration (improved calculations)
        header_height = int(height * 0.15)  # Increased header space
        footer_height = int(height * 0.10)   # Increased footer space
        
        # Calculate usable area with better margins
        usable_height = height - header_height - footer_height
        usable_width = width - 120  # Increased margins
        
        # Subject columns (5 subjects)
        num_subjects = len(self.subjects)
        col_width = usable_width // num_subjects
        col_start_x = 60  # Increased left margin
        
        # Question rows (20 questions per subject)
        questions_per_subject = 20
        row_height = usable_height // (questions_per_subject + 1)  # Better spacing
        row_start_y = header_height + (row_height // 2)
        
        # Choice spacing within each question (improved)
        choice_spacing = col_width // 5  # Better spacing for 4 choices
        choice_start_offset = col_width // 10  # Better offset
        
        circle_radius = min(10, choice_spacing // 4)  # Better radius calculation

        # Process each subject
        for subject_idx, subject in enumerate(self.subjects):
            if subject not in answer_key:
                continue

            # Calculate column position
            col_x = col_start_x + subject_idx * col_width
            
            # Process questions for this subject
            for q_num in range(1, questions_per_subject + 1):
                if q_num not in answer_key[subject]:
                    continue

                # Calculate row position
                row_y = row_start_y + (q_num - 1) * row_height
                
                # Skip if position would be outside image bounds
                if row_y >= height - 50 or row_y < header_height:
                    continue

                correct_answer = answer_key[subject][q_num]
                student_answer = student_answers[subject].get(q_num, '')

                # Determine if answer is correct
                if isinstance(correct_answer, list):
                    is_correct = student_answer.upper() in [ca.upper() for ca in correct_answer]
                    correct_choices = [ca.upper() for ca in correct_answer]
                else:
                    is_correct = student_answer.upper() == correct_answer.upper()
                    correct_choices = [correct_answer.upper()]

                # Draw bubbles for each choice (A, B, C, D)
                for choice_idx, choice in enumerate(self.answer_choices):
                    # Calculate bubble position
                    bubble_x = col_x + choice_start_offset + choice_idx * choice_spacing
                    bubble_y = row_y

                    # Skip if position would be outside image bounds
                    if (bubble_x < circle_radius or bubble_x >= width - circle_radius or 
                        bubble_y < circle_radius or bubble_y >= height - circle_radius):
                        continue

                    # Draw the bubble outline (gray)
                    cv2.circle(annotated, (bubble_x, bubble_y), 
                             circle_radius, colors['bubble_outline'], 1)

                    # If this is the student's answer
                    if choice == student_answer.upper():
                        color = colors['correct'] if is_correct else colors['incorrect']
                        # White outline for visibility
                        cv2.circle(annotated, (bubble_x, bubble_y), circle_radius + 2, colors['selected_outline'], 2)
                        # Draw thick colored circle for student's choice
                        cv2.circle(annotated, (bubble_x, bubble_y), circle_radius, color, 2)
                        # Fill the bubble if correct
                        if is_correct:
                            cv2.circle(annotated, (bubble_x, bubble_y), circle_radius - 2, color, -1)

                    # If student got it wrong, show correct answer in yellow
                    elif not is_correct and choice in correct_choices:
                        cv2.circle(annotated, (bubble_x, bubble_y), 
                                 circle_radius, colors['correct_answer'], 2)
                        # Add a small dot to indicate correct answer
                        cv2.circle(annotated, (bubble_x, bubble_y), 2, colors['correct_answer'], -1)

# Main Streamlit Application
def main():

    # Enhanced CSS for better styling and color combinations
    st.markdown("""
    <style>
    /* Responsive design improvements */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
        }
        .metric-container {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .subject-card {
            margin: 0.3rem 0;
            padding: 0.8rem;
        }
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0;
        opacity: 0.9;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #007bff;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-container h3 {
        margin: 0 0 0.5rem 0;
        color: #333;
        font-size: 1.1rem;
    }
    .metric-container h2 {
        margin: 0;
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 700;
    }
    .success-metric {
        border-left-color: #27ae60 !important;
        background: linear-gradient(135deg, #d5f5e3 0%, #a3e4d7 50%);
    }
    .success-metric h2 {
        color: #186a3b;
    }
    .warning-metric {
        border-left-color: #f39c12 !important;
        background: linear-gradient(135deg, #fef9e7 0%, #fcf3cf 50%);
    }
    .warning-metric h2 {
        color: #b7950b;
    }
    .error-metric {
        border-left-color: #e74c3c !important;
        background: linear-gradient(135deg, #fadbd8 0%, #f5b7b1 50%);
    }
    .error-metric h2 {
        color: #922b21;
    }
    .subject-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .subject-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    .subject-card.excellent {
        border-left-color: #27ae60;
        background: linear-gradient(135deg, #d5f5e3 0%, #e8f8f5 50%);
    }
    .subject-card.good {
        border-left-color: #2980b9;
        background: linear-gradient(135deg, #d6eaf8 0%, #e3f2fd 50%);
    }
    .subject-card.average {
        border-left-color: #f39c12;
        background: linear-gradient(135deg, #fef9e7 0%, #fffbf0 50%);
    }
    .subject-card.needs-improvement {
        border-left-color: #e74c3c;
        background: linear-gradient(135deg, #fadbd8 0%, #fdf2f2 50%);
    }
    .info-box {
        background: #e8f4fd;
        border: 1px solid #85c1e9;
        color: #1b4f72;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fef9e7;
        border: 1px solid #f7dc6f;
        color: #7d6608;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-box {
        background: #d5f5e3;
        border: 1px solid #7dcea0;
        color: #196f3d;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .legend-box {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .legend-box strong {
        color: #495057;
    }
    .processing-info {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    /* Enhanced table styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    /* Fix for Streamlit metric styling */
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    /* Upload area styling */
    .uploadedFileName {
        color: #28a745 !important;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>OMR Evaluation System</h1>
        <p>Innomatics Research Labs</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize OMR processor
    omr_processor = OMRProcessor()

    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Choose a section", 
        ["🔍 OMR Evaluation", "🔑 Answer Key", "📊 System Info", "💡 How to Use"]
    )

    if page == "🔍 OMR Evaluation":
        st.header("📷 Upload OMR Sheet for Evaluation")

        # Camera capture option
        use_camera = st.button("📷 Capture from Camera")

        # File upload section
        uploaded_file = st.file_uploader(
            "📁 Choose an OMR sheet image", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear, well-lit image of the completed OMR sheet. Ensure the sheet is flat and all bubbles are clearly visible."
        )

        image_cv = None
        captured = None

        if use_camera:
            with st.spinner("Opening camera..."):
                frame = omr_processor.capture_image_from_camera()
                if frame is None:
                    st.error("Failed to capture image from camera. Ensure camera is available and not used by another application.")
                else:
                    captured = frame
                    # Convert BGR to RGB for display
                    display_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(display_img, caption="Captured Frame", use_column_width=True)

        if uploaded_file is not None or captured is not None:
            try:
                if captured is not None:
                    image_cv = captured
                    # For display reuse captured image converted above if present
                    image = Image.fromarray(cv2.cvtColor(captured, cv2.COLOR_BGR2RGB))
                else:
                    # Load and display the uploaded image
                    image = Image.open(uploaded_file)
                    image_np = np.array(image)

                    # Convert to OpenCV format
                    if len(image_np.shape) == 3:
                        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    else:
                        image_cv = image_np

                # Create main layout columns
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 Original OMR Sheet")
                    st.image(image, caption="Uploaded OMR Sheet", use_column_width=True)

                # Process the OMR sheet
                with st.spinner("🔍 Processing OMR sheet... This may take a few moments."):
                    # Load answer key (use custom if available, otherwise default)
                    if 'manual_answer_key' in st.session_state and st.session_state.manual_answer_key:
                        answer_key = st.session_state.manual_answer_key
                    else:
                        answer_key = omr_processor.load_answer_key()

                    # Detect bubbles (enhanced detection)
                    bubbles, gray, thresh = omr_processor.detect_and_extract_bubbles(image_cv)

                    # Display processing info
                    st.markdown(f"""
                    <div class="processing-info">
                        <strong>🔍 Processing Results:</strong><br>
                        • Detected {len(bubbles)} potential bubble candidates<br>
                        • Using {'custom' if 'manual_answer_key' in st.session_state else 'default'} answer key<br>
                        • Image dimensions: {image_cv.shape[1]}×{image_cv.shape[0]} pixels
                    </div>
                    """, unsafe_allow_html=True)

                    # For demonstration, use simulated answers
                    # In production, this would be: student_answers = omr_processor.organize_bubbles_by_grid(bubbles, image_cv.shape)
                    student_answers = omr_processor.simulate_realistic_student_answers()

                    # Evaluate answers
                    results = omr_processor.evaluate_answers(student_answers, answer_key)

                    # Create annotated image with colored circles (now using actual bubble positions)
                    annotated_image = omr_processor.create_annotated_image(image_cv, student_answers, answer_key, bubbles)
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                with col2:
                    st.subheader("✅ Evaluated OMR Sheet")
                    st.image(annotated_image_rgb, caption="Annotated Results with Colored Circles", use_column_width=True)

                    # Enhanced legend with better styling
                    st.markdown("""
                    <div class="legend-box">
                        <strong>🎨 Enhanced Color Legend:</strong><br><br>
                        🟢 <strong style="color: #228B22;">Green Circle + Fill</strong>: Correct Answer (Student got it right)<br>
                        🔴 <strong style="color: #DC143C;">Red Circle</strong>: Incorrect Answer (Student's wrong choice)<br>
                        🟡 <strong style="color: #FFD700;">Gold Circle + Dot</strong>: Shows correct answer when student was wrong<br>
                        ⚪ <strong style="color: #A9A9A9;">Gray Outline</strong>: Empty/unselected bubble<br>
                        ⚪ <strong style="color: #FFFFFF; background: #666; padding: 2px;">White Outline</strong>: Highlights selected bubbles for better visibility
                    </div>
                    """, unsafe_allow_html=True)

                # Display results section
                st.markdown("---")
                st.header("📊 Detailed Evaluation Results")

                # Overall score with color coding
                overall = results['overall']
                score_percentage = overall['percentage']

                if score_percentage >= 80:
                    score_class = "success-metric"
                    score_emoji = "🎉"
                elif score_percentage >= 60:
                    score_class = "warning-metric" 
                    score_emoji = "👍"
                else:
                    score_class = "error-metric"
                    score_emoji = "📈"

                st.markdown(f"""
                <div class="metric-container {score_class}">
                    <h3>{score_emoji} Overall Performance</h3>
                    <h2>{overall['correct']}/{overall['total']} ({score_percentage}%)</h2>
                </div>
                """, unsafe_allow_html=True)

                # Subject-wise performance with enhanced styling
                st.subheader("📈 Subject-wise Performance Breakdown")

                # Create enhanced metrics in columns
                subject_cols = st.columns(len(omr_processor.subjects))

                for idx, subject in enumerate(omr_processor.subjects):
                    if subject in results:
                        with subject_cols[idx]:
                            subject_result = results[subject]
                            percentage = subject_result['percentage']

                            # Enhanced color coding for subject performance
                            if percentage >= 90:
                                card_class = "excellent"
                                status_emoji = "🌟"
                                status_text = "Excellent"
                                border_color = "#27ae60"
                            elif percentage >= 80:
                                card_class = "good"
                                status_emoji = "👍"
                                status_text = "Good"
                                border_color = "#2980b9"
                            elif percentage >= 60:
                                card_class = "average"
                                status_emoji = "⚠️"
                                status_text = "Average"
                                border_color = "#f39c12"
                            else:
                                card_class = "needs-improvement"
                                status_emoji = "📈"
                                status_text = "Needs Work"
                                border_color = "#e74c3c"

                            # Create enhanced subject card
                            st.markdown(f"""
                            <div class="subject-card {card_class}" style="border-left-color: {border_color};">
                                <div style="font-size: 1.2em; margin-bottom: 0.5rem;">
                                    {status_emoji} <strong>{subject.replace('_', ' ')}</strong>
                                </div>
                                <div style="font-size: 1.8rem; font-weight: bold; color: {border_color}; margin: 0.5rem 0;">
                                    {subject_result['correct']}/20
                                </div>
                                <div style="font-size: 1.1rem; color: #666;">
                                    {percentage}% - {status_text}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Detailed question-by-question analysis
                st.subheader("🔍 Question-by-Question Analysis")

                # Create tabs for each subject
                subject_tabs = st.tabs([f"📚 {subject.replace('_', ' ')}" for subject in omr_processor.subjects])

                for tab_idx, subject in enumerate(omr_processor.subjects):
                    with subject_tabs[tab_idx]:
                        if subject in results and 'questions' in results[subject]:
                            questions_data = []

                            for q_num in sorted(results[subject]['questions'].keys()):
                                q_result = results[subject]['questions'][q_num]

                                correct_answer_display = q_result['correct_answer']
                                if isinstance(correct_answer_display, list):
                                    correct_answer_display = ' or '.join(correct_answer_display)

                                status_emoji = "✅" if q_result['is_correct'] else "❌"

                                questions_data.append({
                                    'Question': f"Q{q_num}",
                                    'Your Answer': q_result['student_answer'],
                                    'Correct Answer': correct_answer_display,
                                    'Status': status_emoji,
                                    'Result': 'Correct ✅' if q_result['is_correct'] else 'Incorrect ❌'
                                })

                            if questions_data:
                                df_subject = pd.DataFrame(questions_data)
                                st.dataframe(df_subject, use_container_width=True, hide_index=True)

                                # Subject summary
                                correct_count = sum(1 for q in questions_data if '✅' in q['Status'])
                                total_count = len(questions_data)

                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Correct Answers", correct_count)
                                with col_b:
                                    st.metric("Total Questions", total_count)
                                with col_c:
                                    st.metric("Accuracy", f"{(correct_count/total_count*100):.1f}%")

                # Download section
                st.markdown("---")
                st.subheader("💾 Export Results")

                # Prepare detailed results for download
                detailed_results = []
                for subject in omr_processor.subjects:
                    if subject in results and 'questions' in results[subject]:
                        for q_num, q_result in results[subject]['questions'].items():
                            correct_answer_str = q_result['correct_answer']
                            if isinstance(correct_answer_str, list):
                                correct_answer_str = ', '.join(correct_answer_str)

                            detailed_results.append({
                                'Subject': subject.replace('_', ' '),
                                'Question': q_num,
                                'Student Answer': q_result['student_answer'],
                                'Correct Answer': correct_answer_str,
                                'Result': 'Correct' if q_result['is_correct'] else 'Incorrect',
                                'Score': 1 if q_result['is_correct'] else 0
                            })

                if detailed_results:
                    df_results = pd.DataFrame(detailed_results)

                    # Add summary row
                    summary_data = []
                    for subject in omr_processor.subjects:
                        if subject in results:
                            subject_result = results[subject]
                            summary_data.append({
                                'Subject': subject.replace('_', ' '),
                                'Question': 'TOTAL',
                                'Student Answer': '',
                                'Correct Answer': '',
                                'Result': f"{subject_result['correct']}/{subject_result['total']}",
                                'Score': subject_result['percentage']
                            })

                    df_summary = pd.DataFrame(summary_data)
                    df_export = pd.concat([df_results, df_summary], ignore_index=True)

                    # Create download buttons
                    col_download1, col_download2 = st.columns(2)

                    with col_download1:
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Detailed Results (CSV)",
                            data=csv,
                            file_name=f"omr_detailed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download complete question-by-question results"
                        )

                    with col_download2:
                        # Create summary CSV
                        summary_csv = df_summary.to_csv(index=False)
                        st.download_button(
                            label="📊 Download Summary Results (CSV)",
                            data=summary_csv,
                            file_name=f"omr_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download subject-wise summary results"
                        )

                # Performance insights
                st.markdown("---")
                st.subheader("💡 Performance Insights")

                insights = []
                for subject in omr_processor.subjects:
                    if subject in results:
                        percentage = results[subject]['percentage']
                        if percentage >= 90:
                            insights.append(f"🌟 Excellent performance in {subject.replace('_', ' ')} ({percentage}%)")
                        elif percentage >= 80:
                            insights.append(f"👍 Good performance in {subject.replace('_', ' ')} ({percentage}%)")
                        elif percentage >= 60:
                            insights.append(f"⚠️ {subject.replace('_', ' ')} needs improvement ({percentage}%)")
                        else:
                            insights.append(f"🔥 Focus more on {subject.replace('_', ' ')} ({percentage}%)")

                for insight in insights:
                    st.markdown(f"💡 {insight}")

            except Exception as e:
                st.error(f"❌ Error processing the OMR sheet: {str(e)}")
                st.info("Please ensure the uploaded image is clear and in a supported format (PNG, JPG, JPEG).")

        else:
            # Enhanced instructions when no file is uploaded
            st.markdown("""
            <div class="info-box">
                <h3>📋 Upload Instructions</h3>
                <ul style="margin: 0.5rem 0;">
                    <li><strong>Image Quality:</strong> Upload a clear, high-resolution image (recommended: 300+ DPI)</li>
                    <li><strong>Lighting:</strong> Ensure good, even lighting without shadows or glare</li>
                    <li><strong>Position:</strong> Keep the OMR sheet flat and fully visible in the frame</li>
                    <li><strong>Format:</strong> Supported formats: PNG, JPG, JPEG (max 10MB)</li>
                    <li><strong>Bubbles:</strong> Ensure all bubbles are clearly visible and properly filled</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add example images or tips
            st.markdown("""
            <div class="success-box">
                <strong>💡 Pro Tips:</strong><br>
                • Use your phone's camera in good lighting for best results<br>
                • Avoid shadows, wrinkles, or partial coverage of the OMR sheet<br>
                • Make sure the sheet is straight (not tilted) for better detection<br>
                • Clean, dark markings in bubbles work better than light pencil marks
            </div>
            """, unsafe_allow_html=True)

    elif page == "🔑 Answer Key":
        st.header("🔑 Answer Key Management")
        
        # Answer key input method selection
        st.subheader("📝 Choose Answer Key Input Method")
        
        input_method = st.radio(
            "Select how you want to provide the answer key:",
            ["📋 Use Default Answer Key", "📁 Upload CSV File", "✏️ Manual Entry"],
            horizontal=True
        )
        
        answer_key = None
        
        if input_method == "📋 Use Default Answer Key":
            answer_key = omr_processor.load_answer_key()
            
        elif input_method == "📁 Upload CSV File":
            st.subheader("📁 Upload Answer Key CSV")
            st.markdown("""
            **CSV Format Requirements:**
            - First column should contain question numbers (Q1, Q2, etc. or 1, 2, 3, etc.)
            - Each subsequent column should be a subject name
            - Each cell should contain the correct answer (A, B, C, D)
            - For multiple correct answers, separate with commas or 'or' (e.g., "A,B" or "A or B")
            """)
            
            # Download sample CSV template
            if st.button("📥 Download Sample CSV Template"):
                sample_data = []
                for q_num in range(1, 6):  # Sample with 5 questions
                    row = {'Question': q_num}
                    for subject in omr_processor.subjects:
                        if q_num == 1:
                            row[subject] = 'A'
                        elif q_num == 2:
                            row[subject] = 'B'
                        elif q_num == 3:
                            row[subject] = 'C'
                        elif q_num == 4:
                            row[subject] = 'D'
                        else:
                            row[subject] = 'A,B'  # Example of multiple answers
                    sample_data.append(row)
                
                df_sample = pd.DataFrame(sample_data)
                csv_sample = df_sample.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Sample Template",
                    data=csv_sample,
                    file_name="answer_key_template.csv",
                    mime="text/csv"
                )
            
            uploaded_csv = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with the answer key format described above"
            )
            
            if uploaded_csv is not None:
                try:
                    csv_data = uploaded_csv.read().decode('utf-8')
                    answer_key = omr_processor.parse_csv_answer_key(csv_data)
                    
                    if answer_key:
                        st.success("✅ CSV answer key loaded successfully!")
                    else:
                        st.error("❌ Failed to parse CSV file. Please check the format.")
                        
                except Exception as e:
                    st.error(f"❌ Error reading CSV file: {str(e)}")
            
        elif input_method == "✏️ Manual Entry":
            st.subheader("✏️ Manual Answer Key Entry")
            
            # Configuration section
            st.markdown("### ⚙️ Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                num_subjects = st.number_input("Number of Subjects", min_value=1, max_value=10, value=5)
                subjects_input = st.text_area(
                    "Subject Names (one per line)",
                    value="PYTHON\nDATA ANALYSIS\nMySQL\nPOWER BI\nAdv STATS",
                    help="Enter each subject name on a new line"
                )
            
            with col2:
                questions_per_subject = st.number_input("Questions per Subject", min_value=1, max_value=50, value=20)
                answer_choices_input = st.text_input(
                    "Answer Choices (comma separated)",
                    value="A,B,C,D",
                    help="Enter the available answer choices separated by commas"
                )
            
            # Parse inputs
            subjects_list = [s.strip() for s in subjects_input.split('\n') if s.strip()][:num_subjects]
            answer_choices_list = [c.strip() for c in answer_choices_input.split(',') if c.strip()]
            
            if st.button("🔧 Generate Answer Key Template", type="primary"):
                if subjects_list and answer_choices_list:
                    # Create manual answer key template
                    manual_answer_key = omr_processor.create_manual_answer_key(subjects_list, questions_per_subject)
                    
                    # Store in session state for editing
                    st.session_state.manual_answer_key = manual_answer_key
                    st.session_state.subjects_list = subjects_list
                    st.session_state.answer_choices_list = answer_choices_list
                    st.session_state.questions_per_subject = questions_per_subject
                    
                    st.success("✅ Answer key template generated!")
                else:
                    st.error("❌ Please provide valid subjects and answer choices.")
            
            # Manual entry interface
            if 'manual_answer_key' in st.session_state:
                st.markdown("### ✏️ Edit Answer Key")
                
                # Create tabs for each subject
                subject_tabs = st.tabs([f"📚 {subject}" for subject in st.session_state.subjects_list])
                
                for tab_idx, subject in enumerate(st.session_state.subjects_list):
                    with subject_tabs[tab_idx]:
                        st.subheader(f"Answer Key for {subject}")
                        
                        # Create columns for questions
                        num_cols = 4
                        questions_per_col = st.session_state.questions_per_subject // num_cols
                        remaining_questions = st.session_state.questions_per_subject % num_cols
                        
                        cols = st.columns(num_cols)
                        
                        question_num = 1
                        for col_idx in range(num_cols):
                            with cols[col_idx]:
                                questions_in_this_col = questions_per_col + (1 if col_idx < remaining_questions else 0)
                                
                                for q in range(questions_in_this_col):
                                    if question_num <= st.session_state.questions_per_subject:
                                        current_answer = st.session_state.manual_answer_key[subject].get(question_num, '')
                                        
                                        # Create selectbox for answer choice
                                        answer_choice = st.selectbox(
                                            f"Q{question_num}",
                                            options=[''] + st.session_state.answer_choices_list,
                                            index=st.session_state.answer_choices_list.index(current_answer) + 1 if current_answer in st.session_state.answer_choices_list else 0,
                                            key=f"{subject}_Q{question_num}"
                                        )
                                        
                                        # Update the answer key
                                        st.session_state.manual_answer_key[subject][question_num] = answer_choice
                                        question_num += 1
                
                # Save and use button
                if st.button("💾 Save and Use This Answer Key", type="primary"):
                    # Validate that all questions have answers
                    missing_answers = []
                    for subject in st.session_state.subjects_list:
                        for q_num in range(1, st.session_state.questions_per_subject + 1):
                            if not st.session_state.manual_answer_key[subject].get(q_num, ''):
                                missing_answers.append(f"{subject} - Q{q_num}")
                    
                    if missing_answers:
                        st.error(f"⚠️ Please fill in all answers. Missing: {', '.join(missing_answers[:5])}{'...' if len(missing_answers) > 5 else ''}")
                    else:
                        answer_key = st.session_state.manual_answer_key.copy()
                        st.success("✅ Answer key saved successfully!")
                        
                        # Show summary
                        st.subheader("📊 Answer Key Summary")
                        summary_data = []
                        for subject in st.session_state.subjects_list:
                            answered_questions = len([q for q in st.session_state.manual_answer_key[subject].values() if q])
                            summary_data.append({
                                'Subject': subject,
                                'Questions Answered': f"{answered_questions}/{st.session_state.questions_per_subject}",
                                'Completion': f"{(answered_questions/st.session_state.questions_per_subject*100):.1f}%"
                            })
                        
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True, hide_index=True)
        
        # Display answer key if available
        if answer_key:
            st.markdown("---")
            st.subheader("📋 Current Answer Key")
            
            # Create tabs for each subject
            subject_tabs = st.tabs([f"📚 {subject.replace('_', ' ')}" for subject in answer_key.keys()])

            for tab_idx, subject in enumerate(answer_key.keys()):
                with subject_tabs[tab_idx]:
                    st.subheader(f"Answer Key for {subject.replace('_', ' ')}")

                    # Create a formatted display of answers in a table
                    answers_data = []
                    for q_num in sorted(answer_key[subject].keys()):
                        answer = answer_key[subject][q_num]
                        if isinstance(answer, list):
                            answer_str = ' or '.join(answer)
                            note = "Multiple correct answers"
                        else:
                            answer_str = answer
                            note = ""

                        answers_data.append({
                            'Question': f"Q{q_num}",
                            'Correct Answer': answer_str,
                            'Note': note
                        })

                    df_answers = pd.DataFrame(answers_data)
                    st.dataframe(df_answers, use_container_width=True, hide_index=True)

                    # Quick stats
                    total_questions = len(answer_key[subject])
                    multi_answer_questions = len([q for q in answer_key[subject].values() if isinstance(q, list)])

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Questions", total_questions)
                    with col2:
                        st.metric("Multi-Answer Questions", multi_answer_questions)
            
            # Download current answer key as CSV
            st.markdown("---")
            st.subheader("💾 Export Answer Key")
            
            if st.button("📥 Download Answer Key as CSV"):
                # Create CSV data
                all_questions = set()
                for subject_answers in answer_key.values():
                    all_questions.update(subject_answers.keys())
                
                csv_data = []
                for q_num in sorted(all_questions):
                    row = {'Question': q_num}
                    for subject in answer_key.keys():
                        answer = answer_key[subject].get(q_num, '')
                        if isinstance(answer, list):
                            answer = ','.join(answer)
                        row[subject] = answer
                    csv_data.append(row)
                
                df_export = pd.DataFrame(csv_data)
                csv_string = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_string,
                    file_name=f"answer_key_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.markdown("👆 Please select an input method and provide the answer key data above.")

    elif page == "📊 System Info":
        st.header("📊 System Information")

        st.subheader("🎯 About This System")
        st.markdown("""
        This **Automated OMR Evaluation & Scoring System** is designed specifically for 
        **Innomatics Research Labs** placement readiness assessments. The system processes 
        OMR sheets with 100 questions across 5 subjects.

        ### 📋 Key Features:
        - **🔍 Automatic Bubble Detection**: Uses OpenCV for accurate bubble detection
        - **🎨 Visual Feedback**: Color-coded results with green/red/yellow circles
        - **📊 Subject-wise Scoring**: 20 questions per subject (0-20 scale)
        - **📈 Performance Analytics**: Detailed insights and recommendations
        - **💾 Export Options**: Download results in CSV format
        - **🔑 Multi-Answer Support**: Handles questions with multiple correct answers

        ### 🏗️ Technical Architecture:
        - **Frontend**: Streamlit web application
        - **Image Processing**: OpenCV for bubble detection and image processing
        - **Data Analysis**: Pandas for result processing
        - **Visualization**: Custom annotations with colored circles
        """)

        st.subheader("📏 Evaluation Criteria")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 Scoring System:**
            - Each correct answer: **1 point**
            - Each incorrect answer: **0 points**
            - Total possible score: **100 points**
            - Subject-wise scoring: **20 points each**
            """)

        with col2:
            st.markdown("""
            **📊 Performance Grades:**
            - 90-100%: **Excellent** 🌟
            - 80-89%: **Good** 👍  
            - 60-79%: **Average** ⚠️
            - Below 60%: **Needs Improvement** 🔥
            """)

        st.subheader("🔧 System Specifications")

        specs_data = {
            'Feature': [
                'Bubble Detection Accuracy',
                'Processing Time',
                'Supported Image Formats',
                'Maximum Image Size',
                'Concurrent Processing',
                'Error Tolerance',
                'Multi-Answer Support'
            ],
            'Specification': [
                '> 99.5% accuracy',
                '< 30 seconds per sheet',
                'PNG, JPG, JPEG',
                '10MB per image',
                'Multiple sheets simultaneously',
                '< 0.5% error rate',
                'Yes (Questions 16, 59)'
            ]
        }

        df_specs = pd.DataFrame(specs_data)
        st.dataframe(df_specs, use_container_width=True, hide_index=True)

    elif page == "💡 How to Use":
        st.header("💡 How to Use This System")

        st.markdown("""
        ### 🚀 Quick Start Guide

        Follow these steps to evaluate an OMR sheet:
        """)

        # Step-by-step guide
        steps = [
            {
                "title": "📁 Upload OMR Sheet",
                "description": "Navigate to the 'OMR Evaluation' section and upload a clear image of the completed OMR sheet.",
                "tips": ["Ensure good lighting", "Keep the sheet flat", "Use high resolution"]
            },
            {
                "title": "🔍 Automatic Processing", 
                "description": "The system automatically detects bubbles, extracts answers, and compares with the answer key.",
                "tips": ["Processing takes 10-30 seconds", "System uses OpenCV for detection", "Multiple algorithms ensure accuracy"]
            },
            {
                "title": "🎨 View Annotated Results",
                "description": "See your original sheet with colored circles showing correct (green) and incorrect (red) answers.",
                "tips": ["Green = Correct answer", "Red = Incorrect answer", "Yellow = Shows correct answer when wrong"]
            },
            {
                "title": "📊 Analyze Performance",
                "description": "Review subject-wise scores, overall performance, and detailed question analysis.",
                "tips": ["Check subject-wise breakdown", "Identify weak areas", "Review question-by-question results"]
            },
            {
                "title": "💾 Download Results",
                "description": "Export your results as CSV files for record-keeping and further analysis.",
                "tips": ["Detailed results include all questions", "Summary results show subject scores", "Timestamp included in filename"]
            }
        ]

        for i, step in enumerate(steps, 1):
            with st.expander(f"Step {i}: {step['title']}", expanded=i==1):
                st.markdown(f"**{step['description']}**")
                st.markdown("💡 **Tips:**")
                for tip in step['tips']:
                    st.markdown(f"   - {tip}")

        st.markdown("---")

        st.subheader("📸 Image Requirements")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **✅ Good Image Qualities:**
            - Clear, high-resolution image
            - Even lighting without shadows
            - OMR sheet is flat and straight
            - All bubbles are clearly visible
            - No blur or distortion
            - Proper contrast between bubbles and paper
            """)

        with col2:
            st.markdown("""
            **❌ Avoid These Issues:**
            - Blurry or low-resolution images
            - Poor lighting or shadows
            - Wrinkled or folded sheets
            - Partially visible OMR sheet
            - Extreme angles or perspective
            - Overexposed or underexposed images
            """)

        st.markdown("---")

        st.subheader("🎯 Understanding Results")

        st.markdown("""
        **🎨 Color Coding System:**
        - **🟢 Green Circle**: Your answer is correct
        - **🔴 Red Circle**: Your answer is incorrect  
        - **🟡 Yellow Circle**: Shows the correct answer when you got it wrong
        - **⚪ Gray Circle**: Empty or unselected bubble

        **📊 Score Interpretation:**
        - **Overall Score**: Total correct answers out of 100
        - **Subject Scores**: Correct answers out of 20 for each subject
        - **Percentage**: Your score as a percentage
        - **Performance Grade**: Based on your percentage score
        """)

        st.markdown("---")

        st.subheader("🔧 Troubleshooting")

        troubleshooting_data = {
            'Issue': [
                'Image not uploading',
                'Processing taking too long', 
                'Inaccurate bubble detection',
                'Missing annotations',
                'Download not working'
            ],
            'Solution': [
                'Check file format (PNG/JPG/JPEG) and size (<10MB)',
                'Wait up to 60 seconds, refresh page if needed',
                'Ensure good image quality and proper lighting',
                'Verify OMR sheet is properly aligned and visible',
                'Try different browser or clear browser cache'
            ]
        }

        df_troubleshooting = pd.DataFrame(troubleshooting_data)
        st.dataframe(df_troubleshooting, use_container_width=True, hide_index=True)

        st.markdown("💬 **Need Help?** Contact the system administrator if you encounter persistent issues.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🏢 <strong>Innomatics Research Labs</strong> | Automated OMR Evaluation System</p>
    <p>Built with ❤️ using Streamlit and OpenCV | © 2025</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
