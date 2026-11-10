"""Main application file for the Style Finder Gradio interface."""

import os
from tempfile import NamedTemporaryFile
import gradio as gr
import pandas as pd
from PIL import Image

import config
from models.image_processor import ImageProcessor
from models.llm_service import LlamaVisionService
from utils.helpers import (
    format_alternatives_response,
    get_all_items_for_image,
    process_response,
)


class StyleFinderApp:
    def __init__(self, dataset_path: str):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        self.data = pd.read_pickle(dataset_path)
        if self.data.empty:
            raise ValueError("The loaded dataset is empty")

        self.image_processor = ImageProcessor(
            image_size=config.IMAGE_SIZE,
            norm_mean=config.NORMALIZATION_MEAN,
            norm_std=config.NORMALIZATION_STD,
        )
        self.llm_service = LlamaVisionService(
			model_id=config.LLAMA_MODEL_ID,
    		project_id=config.PROJECT_ID,
    		region=config.REGION,
    		api_key=config.API_KEY,
        )

    def process_image(self, image):
        if image is None:
            return "Please upload an image or select an example first."

        image_path = None
        is_temp = False

        try:
            if not isinstance(image, str):
                temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
                image_path = temp_file.name
                temp_file.close()
                image.save(image_path)
                is_temp = True
            else:
                image_path = image

            user_encoding = self.image_processor.encode_image(image_path, is_url=False)
            if user_encoding.get("vector") is None:
                return "Error: Unable to process the image. Please try another image."

            closest_row, similarity_score = self.image_processor.find_closest_match(
                user_encoding["vector"], self.data
            )
            if closest_row is None:
                return "Error: Unable to find a match. Please try another image."

            print(
                f"Closest match: {closest_row.get('Item Name', 'Unknown')} "
                f"with similarity score {similarity_score:.2f}"
            )

            all_items = get_all_items_for_image(closest_row["Image URL"], self.data)
            if all_items.empty:
                return "Error: No items found for the matched image."

            bot_response = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding["base64"],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score,
                threshold=config.SIMILARITY_THRESHOLD,
            )

            return process_response(bot_response)

        finally:
            if is_temp and image_path and os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except OSError:
                    pass


def load_example_image(path: str):
    if os.path.exists(path):
        return Image.open(path)
    return None


def create_gradio_interface(app: StyleFinderApp) -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Style Analyzer") as demo:
        gr.Markdown(
            """
            # Fashion Style Analyzer
            Upload an image to analyze fashion elements and get detailed information about the items.
            """
        )

        gr.Markdown("### Example Images")
        with gr.Row():
            gr.Image(value="examples/test-1.png", label="Example 1", show_label=True, scale=1, interactive=False)
            gr.Image(value="examples/test-2.png", label="Example 2", show_label=True, scale=1, interactive=False)
            gr.Image(value="examples/test-3.png", label="Example 3", show_label=True, scale=1, interactive=False)

        with gr.Row():
            example1_btn = gr.Button("Use Example 1")
            example2_btn = gr.Button("Use Example 2")
            example3_btn = gr.Button("Use Example 3")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Fashion Image")
                submit_btn = gr.Button("Analyze Style", variant="primary")
                status = gr.Markdown("Ready to analyze.")
            with gr.Column(scale=2):
                output = gr.Markdown(label="Style Analysis Results", height=700)

        submit_btn.click(
            fn=lambda: "Analyzing image... This may take a few moments.",
            inputs=None,
            outputs=status,
        ).then(
            fn=app.process_image, inputs=[image_input], outputs=output
        ).then(
            fn=lambda: "Analysis complete!", inputs=None, outputs=status
        )

        example1_btn.click(
            fn=lambda: load_example_image("examples/test-1.png"),
            inputs=None,
            outputs=image_input,
        ).then(
            fn=lambda: "Example 1 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status,
        )

        example2_btn.click(
            fn=lambda: load_example_image("examples/test-2.png"),
            inputs=None,
            outputs=image_input,
        ).then(
            fn=lambda: "Example 2 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status,
        )

        example3_btn.click(
            fn=lambda: load_example_image("examples/test-3.png"),
            inputs=None,
            outputs=image_input,
        ).then(
            fn=lambda: "Example 3 loaded. Click 'Analyze Style' to process.",
            inputs=None,
            outputs=status,
        )

        gr.Markdown(
            """
            ### About This Application
            This system analyzes fashion images using:
            - **Image Encoding:** Converting fashion images into numerical vectors.
            - **Similarity Matching:** Finding visually similar items in a database.
            - **Advanced AI:** Generating detailed descriptions of fashion elements.
            """
        )

    return demo


if __name__ == "__main__":
    try:
        app = StyleFinderApp("swift-style-embeddings.pkl")
        demo = create_gradio_interface(app)
        demo.launch(server_name="127.0.0.1", server_port=5000, share=False)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
