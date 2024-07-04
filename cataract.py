import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import sqlite3
from io import BytesIO
from scipy.stats import norm

# Load YOLO models
try:
    yolo_model_cataract = YOLO('best-cataract-seg.pt')
    yolo_model_object_detection = YOLO('best-cataract-od.pt')
    print("YOLO models loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO models: {e}")

def calculate_ratios(red_values, green_values, blue_values, total_pixels):
    if total_pixels == 0:
        return 0, 0, 0

    red_ratio = np.sum(red_values) / total_pixels
    green_ratio = np.sum(green_values) / total_pixels
    blue_ratio = np.sum(blue_values) / total_pixels

    total_ratio = red_ratio + green_ratio + blue_ratio

    if total_ratio > 0:
        red_quantity = (red_ratio / total_ratio) * 255
        green_quantity = (green_ratio / total_ratio) * 255
        blue_quantity = (blue_ratio / total_ratio) * 255
    else:
        red_quantity, green_quantity, blue_quantity = 0, 0, 0

    return red_quantity, green_quantity, blue_quantity

def cataract_staging(red_quantity, green_quantity, blue_quantity):
    # Assuming you have already defined your mean and std for each class and each RGB channel
    # Example mean and std based on earlier discussion
    mean_mature_red = 73.37
    std_mature_red = (90.12 - 41.49) / 4
    mean_mature_green = 89.48
    std_mature_green = (97.67 - 83.39) / 4
    mean_mature_blue = 92.15
    std_mature_blue = (117.82 - 75.37) / 4
    
    mean_normal_red = 67.84
    std_normal_red = (107.02 - 56.19) / 4
    mean_normal_green = 84.85
    std_normal_green = (89.89 - 80.74) / 4
    mean_normal_blue = 102.31
    std_normal_blue = (111.34 - 65.58) / 4
    
    mean_immature_red = 68.83
    std_immature_red = (85.95 - 41.49) / 4
    mean_immature_green = 89.43
    std_immature_green = (97.67 - 83.39) / 4
    mean_immature_blue = 96.74
    std_immature_blue = (117.82 - 78.41) / 4

    # Calculate likelihoods for each class
    likelihood_mature = (
        norm.pdf(red_quantity, mean_mature_red, std_mature_red) *
        norm.pdf(green_quantity, mean_mature_green, std_mature_green) *
        norm.pdf(blue_quantity, mean_mature_blue, std_mature_blue)
    )
    
    likelihood_normal = (
        norm.pdf(red_quantity, mean_normal_red, std_normal_red) *
        norm.pdf(green_quantity, mean_normal_green, std_normal_green) *
        norm.pdf(blue_quantity, mean_normal_blue, std_normal_blue)
    )
    
    likelihood_immature = (
        norm.pdf(red_quantity, mean_immature_red, std_immature_red) *
        norm.pdf(green_quantity, mean_immature_green, std_immature_green) *
        norm.pdf(blue_quantity, mean_immature_blue, std_immature_blue)
    )

    # Define prior probabilities (assuming equal prior for simplicity)
    prior_mature = 1/3
    prior_normal = 1/3
    prior_immature = 1/3

    # Apply Bayes' theorem to compute posterior probabilities
    posterior_mature = likelihood_mature * prior_mature
    posterior_normal = likelihood_normal * prior_normal
    posterior_immature = likelihood_immature * prior_immature

    # Determine the stage based on maximum posterior probability
    stages = {
        posterior_mature: "Mature",
        posterior_normal: "Normal",
        posterior_immature: "Immature"
    }
    max_posterior = max(posterior_mature, posterior_normal, posterior_immature)
    stage = stages[max_posterior]

    return stage

def add_watermark(image):
    try:
        logo = Image.open('image-logo.png').convert("RGBA")
        image = image.convert("RGBA")
        
        # Resize logo
        basewidth = 100
        wpercent = (basewidth / float(logo.size[0]))
        hsize = int((float(wpercent) * logo.size[1]))
        logo = logo.resize((basewidth, hsize), Image.LANCZOS)
        
        # Position logo
        position = (image.width - logo.width - 10, image.height - logo.height - 10)
        
        # Composite image
        transparent = Image.new('RGBA', (image.width, image.height), (0, 0, 0, 0))
        transparent.paste(image, (0, 0))
        transparent.paste(logo, position, mask=logo)
        
        return transparent.convert("RGB")
    except Exception as e:
        print(f"Error adding watermark: {e}")
        return image


def predict_and_visualize(image):
    try:
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        orig_size = pil_image.size
        results = yolo_model_cataract(pil_image)

        raw_response = str(results)
        masked_image = np.array(pil_image)
        mask_image = np.zeros_like(masked_image)

        red_quantity, green_quantity, blue_quantity = 0, 0, 0
        total_pixels = 0

        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                mask = np.array(result.masks.data.cpu().squeeze().numpy())
                mask_resized = np.array(Image.fromarray(mask).resize(orig_size, Image.NEAREST))

                red_mask = np.zeros_like(masked_image)
                red_mask[mask_resized > 0.5] = [255, 0, 0]
                alpha = 0.5
                blended_image = cv2.addWeighted(masked_image, 1 - alpha, red_mask, alpha, 0)

                pupil_pixels = np.array(pil_image)[mask_resized > 0.5]
                total_pixels = pupil_pixels.shape[0]

                red_values = pupil_pixels[:, 0]
                green_values = pupil_pixels[:, 1]
                blue_values = pupil_pixels[:, 2]

                red_quantity, green_quantity, blue_quantity = calculate_ratios(red_values, green_values, blue_values, total_pixels)
                stage = cataract_staging(red_quantity, green_quantity, blue_quantity)

                # Add text to the blended image
                combined_pil_image = Image.fromarray(blended_image)
                draw = ImageDraw.Draw(combined_pil_image)
                
                # Load a larger font (adjust the size as needed)
                font_size = 48  # Example font size
                try:
                    font = ImageFont.truetype("font.ttf", size=font_size)
                except IOError:
                    font = ImageFont.load_default()
                    print("Error: cannot open resource, using default font.")

                text = f"Red quantity: {red_quantity:.2f}\nGreen quantity: {green_quantity:.2f}\nBlue quantity: {blue_quantity:.2f}\nStage: {stage}"
                
                # Calculate text bounding box
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                text_x = 20
                text_y = 40
                padding = 10

                # Draw a filled rectangle for the background
                draw.rectangle(
                    [text_x - padding, text_y - padding, text_x + text_width + padding, text_y + text_height + padding],
                    fill="black"
                )
                
                # Draw text on top of the rectangle
                draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

                # Add watermark to the image
                combined_pil_image_with_watermark = add_watermark(combined_pil_image)

                return np.array(combined_pil_image_with_watermark), red_quantity, green_quantity, blue_quantity, raw_response, stage

        return image, 0, 0, 0, "No mask detected.", "Unknown"
    
    except Exception as e:
        print("Error:", e)
        return np.zeros_like(image), 0, 0, 0, str(e), "Error"

def check_duplicate_entry(conn, red_quantity, green_quantity, blue_quantity, stage):
    cursor = conn.cursor()
    query = '''SELECT COUNT(*) FROM cataract_results WHERE red_quantity=? AND green_quantity=? AND blue_quantity=? AND stage=?'''
    cursor.execute(query, (red_quantity, green_quantity, blue_quantity, stage))
    count = cursor.fetchone()[0]
    return count > 0

def save_cataract_prediction_to_db(image, red_quantity, green_quantity, blue_quantity, stage):
    database = "cataract_results.db"
    conn = create_connection(database)
    if conn:
        create_cataract_table(conn)
        
        # Check for duplicate entries
        if check_duplicate_entry(conn, red_quantity, green_quantity, blue_quantity, stage):
            conn.close()
            return "Duplicate entry found, not saving.", "Duplicate entry detected."
        
        sql = '''INSERT INTO cataract_results(image, red_quantity, green_quantity, blue_quantity, stage) VALUES(?,?,?,?,?)'''
        cur = conn.cursor()
        
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        
        cur.execute(sql, (img_bytes, red_quantity, green_quantity, blue_quantity, stage))
        conn.commit()
        conn.close()
        return "Data saved successfully", f"Red: {red_quantity}, Green: {green_quantity}, Blue: {blue_quantity}, Stage: {stage}"

    return "Failed to save data", "No connection to the database."

def combined_prediction(image):
    blended_image, red_quantity, green_quantity, blue_quantity, raw_response, stage = predict_and_visualize(image)
    save_message, debug_info = save_cataract_prediction_to_db(Image.fromarray(blended_image), red_quantity, green_quantity, blue_quantity, stage)
    return blended_image, red_quantity, green_quantity, blue_quantity, raw_response, stage, save_message, debug_info

def create_connection(db_file):
    """ Create a database connection to the SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn

def create_cataract_table(conn):
    """ Create the cataract results table if it does not exist """
    create_table_sql = """ CREATE TABLE IF NOT EXISTS cataract_results (
                            id integer PRIMARY KEY,
                            image blob,
                            red_quantity real,
                            green_quantity real,
                            blue_quantity real,
                            stage text
                        ); """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)

def predict_object_detection(image):
    try:
        image_np = np.array(image)
        results = yolo_model_object_detection(image_np)

        image_with_boxes = image_np.copy()
        raw_predictions = []
        for result in results[0].boxes:
            label = "Normal" if result.cls.item() == 1 else "Cataract"
            confidence = result.conf.item()
            xmin, ymin, xmax, ymax = map(int, result.xyxy[0])
            cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            
            font_scale = 1.0
            thickness = 2
            text = f'{label} {confidence:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image_with_boxes, (xmin, ymin - text_height - baseline), (xmin + text_width, ymin), (0, 0, 0), cv2.FILLED)
            cv2.putText(image_with_boxes, text, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            raw_predictions.append(f"Label: {label}, Confidence: {confidence:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

        raw_predictions_str = "\n".join(raw_predictions)

        # Convert image_with_boxes to PIL image and add watermark
        image_with_boxes_pil = Image.fromarray(image_with_boxes)
        image_with_boxes_pil_with_watermark = add_watermark(image_with_boxes_pil)

        return np.array(image_with_boxes_pil_with_watermark), raw_predictions_str
    except Exception as e:
        print("Error in object detection:", e)
        return np.zeros_like(image), str(e)