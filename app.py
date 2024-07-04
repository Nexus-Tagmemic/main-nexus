import gradio as gr
import os
import numpy as np
from cataract import combined_prediction, save_cataract_prediction_to_db, predict_object_detection
from glaucoma import combined_prediction_glaucoma, submit_to_db, predict_image
from database import get_db_data, format_db_data, clear_database
from chatbot import chatbot, toggle_visibility, update_patient_history, generate_voice_response 
from PIL import Image

# Define the custom theme
theme = gr.themes.Soft(
    primary_hue="neutral",
    secondary_hue="neutral",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont('Libre Franklin'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif']
).set(
    body_background_fill="#ffffff",
    block_background_fill="#0a2b42",
    block_border_width="1px",
    block_title_background_fill="#0a2b42",
    input_background_fill="#ffffff",
    button_secondary_background_fill="#0a2b42",
    border_color_primary="#800080",
    background_fill_secondary="#ffffff",
    color_accent_soft="transparent"
)

# Define custom CSS
css = """
body {
    color: #0a2b42;  /* Dark blue font */
}
.light body {
    color: #0a2b42;  /* Dark blue font */
}
input, textarea {
    background-color: #ffffff !important;  /* White background for text boxes */
    color: #0a2b42 !important;  /* Dark blue font for text boxes */
}
"""

logo_url = "https://huggingface.co/spaces/Nexus-Community/nexus-main/resolve/main/Nexus-Hub.png"
db_path_cataract = "cataract_results.db"
db_path_glaucoma = "glaucoma_results.db"

def display_db_data():
    """Fetch and format the data from the database for display."""
    glaucoma_data, cataract_data = get_db_data(db_path_glaucoma, db_path_cataract)
    formatted_data = format_db_data(glaucoma_data, cataract_data)
    return formatted_data

def check_db_status():
    """Check the status of the databases and return a status message."""
    cataract_status = "Loaded" if os.path.exists(db_path_cataract) else "Not Loaded"
    glaucoma_status = "Loaded" if os.path.exists(db_path_glaucoma) else "Not Loaded"
    return f"Cataract Database: {cataract_status}\nGlaucoma Database: {glaucoma_status}"

def toggle_input_visibility(input_type):
    if input_type == "Voice":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)
        
def process_image(image):
    # Run the analyzer model
    blended_image, red_quantity, green_quantity, blue_quantity, raw_response, stage, save_message, debug_info = combined_prediction(image)
    
    # Run the object detection model
    predicted_image_od, raw_response_od = predict_object_detection(image)
    
    return blended_image, red_quantity, green_quantity, blue_quantity, raw_response, stage, save_message, debug_info, predicted_image_od, raw_response_od

with gr.Blocks(theme=theme) as demo:
    gr.HTML(f"<img src='{logo_url}' alt='Logo' width='150'/>")  
    gr.Markdown("## Wellness-Nexus V.1.0")
    gr.Markdown("This app helps people to diagnose their cataract and glaucoma, both respectively #1 and #2 cause of blindness in the world")

    with gr.Tab("Cataract Screener and Analyzer"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload an Image")
            submit_btn = gr.Button("Submit")

        with gr.Row():
            segmented_image_cataract = gr.Image(type="numpy", label="Segmented Image")
            predicted_image_od = gr.Image(type="numpy", label="Predicted Image")
        
        with gr.Column():
            red_quantity_cataract = gr.Slider(label="Red Quantity", minimum=0, maximum=255, interactive=False)
            green_quantity_cataract = gr.Slider(label="Green Quantity", minimum=0, maximum=255, interactive=False)
            blue_quantity_cataract = gr.Slider(label="Blue Quantity", minimum=0, maximum=255, interactive=False)

        with gr.Row():
            cataract_stage = gr.Textbox(label="Cataract Stage", interactive=False)
            raw_response_cataract = gr.Textbox(label="Raw Response", interactive=False)
            submit_value_btn_cataract = gr.Button("Submit Values to Database")
            db_response_cataract = gr.Textbox(label="Database Response")
            debug_cataract = gr.Textbox(label="Debug Message", interactive=False)

        submit_btn.click(
            process_image,
            inputs=image_input,
            outputs=[
                segmented_image_cataract, red_quantity_cataract, green_quantity_cataract, blue_quantity_cataract, raw_response_cataract, cataract_stage, db_response_cataract, debug_cataract, predicted_image_od
            ]
        )

        submit_value_btn_cataract.click(
            lambda img, red, green, blue, stage: save_cataract_prediction_to_db(Image.fromarray(img), red, green, blue, stage),
            inputs=[segmented_image_cataract, red_quantity_cataract, green_quantity_cataract, blue_quantity_cataract, cataract_stage],
            outputs=[db_response_cataract, debug_cataract]
        )

    with gr.Tab("Glaucoma Analyzer and Screener"):
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload an Image")
            mask_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Mask Threshold")
            
        with gr.Row():
            submit_btn_segmentation = gr.Button("Submit Segmentation")
            submit_btn_od = gr.Button("Submit Object Detection")
    
        with gr.Row():
            segmented_image = gr.Image(type="numpy", label="Segmented Image")
            predicted_image_od = gr.Image(type="numpy", label="Predicted Image")
    
        with gr.Row():
            raw_response_od = gr.Textbox(label="Raw Result")
            
        with gr.Column():
            cup_area = gr.Textbox(label="Cup Area")
            disk_area = gr.Textbox(label="Disk Area")
            rim_area = gr.Textbox(label="Rim Area")
            rim_to_disc_ratio = gr.Textbox(label="Rim to Disc Ratio")
            ddls_stage = gr.Textbox(label="DDLS Stage")
            
        with gr.Column():
            submit_value_btn = gr.Button("Submit Values to Database")
            db_response = gr.Textbox(label="Database Response")
            debug_glaucoma = gr.Textbox(label="Debug Message", interactive=False)
    
        def process_segmentation_image(img, mask_thresh):
            # Run the segmentation model
            return combined_prediction_glaucoma(img, mask_thresh)
    
        def process_od_image(img):
            # Run the object detection model
            image_with_boxes, raw_predictions = predict_image(img)
            return image_with_boxes, raw_predictions
    
        submit_btn_segmentation.click(
            fn=process_segmentation_image,
            inputs=[image_input, mask_threshold_slider],
            outputs=[
                segmented_image, cup_area, disk_area, rim_area, rim_to_disc_ratio, ddls_stage
            ]
        )

        submit_btn_od.click(
            fn=process_od_image,
            inputs=[image_input],
            outputs=[
                predicted_image_od, raw_response_od
            ]
        )

        submit_value_btn.click(
            lambda img, cup, disk, rim, ratio, stage: submit_to_db(img, cup, disk, rim, ratio, stage),
            inputs=[image_input, cup_area, disk_area, rim_area, rim_to_disc_ratio, ddls_stage],
            outputs=[db_response, debug_glaucoma]
        )

    with gr.Tab("Chatbot"):
        with gr.Row():
            input_type_dropdown = gr.Dropdown(label="Input Type", choices=["Voice", "Text"], value="Voice")
            tts_model_dropdown = gr.Dropdown(label="TTS Model", choices=["Ryan (ESPnet)", "Nithu (Custom)"], value="Nithu (Custom)")
            submit_btn_chatbot = gr.Button("Submit")

        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Record your voice", visible=True)
            text_input = gr.Textbox(label="Type your question", visible=False)

        with gr.Row():
            answer_textbox = gr.Textbox(label="Answer")
            answer_audio = gr.Audio(label="Answer as Speech", type="filepath")
            generate_voice_btn = gr.Button("Generate Voice Response")

        with gr.Row():
            log_messages_textbox = gr.Textbox(label="Log Messages", lines=10)
            db_status_textbox = gr.Textbox(label="Database Status", interactive=False)

        input_type_dropdown.change(
            fn=toggle_input_visibility,
            inputs=[input_type_dropdown],
            outputs=[audio_input, text_input]
        )

        submit_btn_chatbot.click(
            fn=chatbot,
            inputs=[audio_input, input_type_dropdown, text_input],
            outputs=[answer_textbox, db_status_textbox]
        )

        generate_voice_btn.click(
            fn=generate_voice_response,
            inputs=[tts_model_dropdown, answer_textbox],
            outputs=[answer_audio, db_status_textbox]
        )

        fetch_db_btn = gr.Button("Fetch Database")
        fetch_db_btn.click(
            fn=update_patient_history,
            inputs=[],
            outputs=[db_status_textbox]
        )
        
    with gr.Tab("Database Upload and View"):
        gr.Markdown("### Store and Retrieve Context Information")

        db_display = gr.HTML()
        load_db_btn = gr.Button("Load Database Content")
        load_db_btn.click(display_db_data, outputs=db_display)

        # Buttons to clear databases
        clear_cataract_db_btn = gr.Button("Clear Cataract Database")
        clear_glaucoma_db_btn = gr.Button("Clear Glaucoma Database")

        clear_cataract_db_btn.click(
            fn=clear_database,
            inputs=[gr.State(value=db_path_cataract), gr.State(value="cataract_results")],
            outputs=db_display
        )

        clear_glaucoma_db_btn.click(
            fn=clear_database,
            inputs=[gr.State(value=db_path_glaucoma), gr.State(value="results")],
            outputs=db_display
        )
        
    demo.launch(share=True)