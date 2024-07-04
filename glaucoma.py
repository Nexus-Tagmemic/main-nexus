import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO
from database import save_prediction_to_db

# Load YOLO models
try:
    yolo_model_glaucoma = YOLO('best-glaucoma-seg.pt')
    yolo_model_od = YOLO("best-glaucoma-od.pt")
    print("YOLO models loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO models: {e}")

def calculate_area(mask):
    area = np.sum(mask > 0.5)
    print(f"Calculated area: {area}")
    return area

def classify_ddls(rim_to_disc_ratio):
    if rim_to_disc_ratio >= 0.5:
        stage = 0  # Non Glaucomatous
    elif 0.4 <= rim_to_disc_ratio < 0.5:
        stage = 1
    elif 0.3 <= rim_to_disc_ratio < 0.4:
        stage = 2
    elif 0.2 <= rim_to_disc_ratio < 0.3:
        stage = 3
    elif 0.1 <= rim_to_disc_ratio < 0.2:
        stage = 4
    elif 0.0 < rim_to_disc_ratio < 0.1:
        stage = 5
    else:
        stage = 6
    print(f"Classified DDLS stage: {stage}")
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

def predict_and_visualize_glaucoma(image, mask_threshold=0.5):
    try:
        pil_image = Image.fromarray(image)
        orig_size = pil_image.size
        results = yolo_model_glaucoma(pil_image)

        raw_response = str(results)
        print(f"YOLO results: {raw_response}")
        masked_image = np.array(pil_image)
        mask_image = np.zeros_like(masked_image)

        cup_mask, disk_mask = None, None

        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0:
                for mask_data in result.masks.data:
                    mask = np.array(mask_data.cpu().squeeze().numpy())
                    mask_resized = cv2.resize(mask, orig_size, interpolation=cv2.INTER_NEAREST)

                    if np.sum(mask_resized) > np.sum(disk_mask if disk_mask is not None else 0):
                        cup_mask = disk_mask
                        disk_mask = mask_resized
                    else:
                        cup_mask = mask_resized

        if cup_mask is not None and disk_mask is not None:
            area_cup = calculate_area(cup_mask)
            area_disk = calculate_area(disk_mask)
            rim_area = area_disk - area_cup
            print(f"Area cup: {area_cup}, Area disk: {area_disk}, Rim area: {rim_area}")

            rim_to_disc_ratio = rim_area / area_disk if area_disk > 0 else 0
            print(f"Rim to disc ratio: {rim_to_disc_ratio}")
            ddls_stage = classify_ddls(rim_to_disc_ratio)

            combined_image = np.array(pil_image)

            # Create RGBA version of the original image
            combined_image_rgba = cv2.cvtColor(combined_image, cv2.COLOR_RGB2RGBA)

            # Create transparent masks
            cup_mask_rgba = np.zeros_like(combined_image_rgba)
            cup_mask_rgba[:, :, 0] = 0    # Red channel
            cup_mask_rgba[:, :, 1] = 0    # Green channel
            cup_mask_rgba[:, :, 2] = 255  # Blue channel
            cup_mask_rgba[:, :, 3] = 128  # Alpha channel (50% transparency)

            disk_mask_rgba = np.zeros_like(combined_image_rgba)
            disk_mask_rgba[:, :, 0] = 255  # Red channel
            disk_mask_rgba[:, :, 1] = 0    # Green channel
            disk_mask_rgba[:, :, 2] = 0    # Blue channel
            disk_mask_rgba[:, :, 3] = 128  # Alpha channel (50% transparency)

            # Apply masks to the original image with transparency
            cup_mask_indices = cup_mask > mask_threshold
            disk_mask_indices = disk_mask > mask_threshold

            combined_image_rgba[cup_mask_indices] = (0.5 * combined_image_rgba[cup_mask_indices] + 0.5 * cup_mask_rgba[cup_mask_indices]).astype(np.uint8)
            combined_image_rgba[disk_mask_indices] = (0.5 * combined_image_rgba[disk_mask_indices] + 0.5 * disk_mask_rgba[disk_mask_indices]).astype(np.uint8)

            # Convert to PIL image for drawing
            combined_pil_image = Image.fromarray(combined_image_rgba)

            # Add text to the image
            draw = ImageDraw.Draw(combined_pil_image)
            
            # Load a larger font (adjust the size as needed)
            font_size = 48  # Example font size
            try:
                font = ImageFont.truetype("font.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()
                print("Error: cannot open resource, using default font.")

            text = f"Area cup: {area_cup}\nArea disk: {area_disk}\nRim area: {rim_area}\nRim to disc ratio: {rim_to_disc_ratio:.2f}\nDDLS stage: {ddls_stage}"
            text_x = 20
            text_y = 40

            draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)

            # Add watermark
            combined_pil_image = add_watermark(combined_pil_image)

            return np.array(combined_pil_image), area_cup, area_disk, rim_area, rim_to_disc_ratio, ddls_stage

        print("No detected regions")
        return np.zeros_like(image), 0, 0, 0, 0, "No detected regions"
    except Exception as e:
        print("Error:", e)
        return np.zeros_like(image), 0, 0, 0, 0, str(e)

def combined_prediction_glaucoma(image, mask_threshold):
    segmented_image, cup_area, disk_area, rim_area, rim_to_disc_ratio, ddls_stage = predict_and_visualize_glaucoma(image, mask_threshold)
    print(f"Segmented image: {segmented_image.shape}")
    print(f"Cup area: {cup_area}, Disk area: {disk_area}, Rim area: {rim_area}")
    print(f"Rim to disc ratio: {rim_to_disc_ratio}, DDLS stage: {ddls_stage}")

    return segmented_image, cup_area, disk_area, rim_area, rim_to_disc_ratio, ddls_stage

def submit_to_db(image, cup_area, disk_area, rim_area, rim_to_disc_ratio, ddls_stage):
    try:
        # Convert the image from numpy array to PIL image
        pil_image = Image.fromarray(np.uint8(image))
        save_prediction_to_db(pil_image, cup_area, disk_area, rim_area, rim_to_disc_ratio, ddls_stage)
        return "Values successfully saved to database.", ""
    except Exception as e:
        print(f"Error saving to database: {e}")
        return f"Error saving to database: {e}", ""

def predict_image(input_image):
    # Convert Gradio input image (PIL Image) to numpy array
    image_np = np.array(input_image)

    # Ensure the image is in the correct format
    if len(image_np.shape) == 2:  # grayscale to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:  # RGBA to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Perform prediction
    results = yolo_model_od(image_np)

    # Draw bounding boxes on the image
    image_with_boxes = image_np.copy()
    raw_predictions = []
    for result in results[0].boxes:
        confidence = result.conf.item()  # Convert tensor to standard Python type
        label = "Glaucoma" if confidence > 0.5 else "Normal"  # Set label based on confidence
        
        xmin, ymin, xmax, ymax = map(int, result.xyxy[0])
        
        # Draw black rectangle as background for text
        text = f'{label} {confidence:.2f}'
        font_scale = 1.0  # Increased font scale
        font_thickness = 2  # Increased font thickness
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.rectangle(image_with_boxes, (xmin, ymin - h - baseline), (xmin + w, ymin), (0, 0, 0), -1)
        
        cv2.putText(image_with_boxes, text, (xmin, ymin - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # Draw thicker bounding box
        box_thickness = 3  # Increased box thickness
        cv2.rectangle(image_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), box_thickness)

        raw_predictions.append(f"Label: {label}, Confidence: {confidence:.2f}, Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

    raw_predictions_str = "\n".join(raw_predictions)
    
    # Add watermark to the final image with boxes
    pil_image_with_boxes = Image.fromarray(image_with_boxes)
    pil_image_with_boxes = add_watermark(pil_image_with_boxes)
    image_with_boxes = np.array(pil_image_with_boxes)
    
    return image_with_boxes, raw_predictions_str