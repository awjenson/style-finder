"""
Service for interacting with the Llama 3.2 Vision Instruct model.
"""

import logging
import urllib.parse
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LlamaVisionService:
    """
    Provides methods to interact with the Llama Vision model.
    """

    def __init__(
        self,
        model_id,
        project_id,
        region="us-south",
        temperature=0.2,
        top_p=0.6,
        api_key=None,
        max_tokens=2000,
    ):
        credentials = Credentials(
            url=f"https://{region}.ml.cloud.ibm.com",
            api_key=api_key
        )
        self.client = APIClient(credentials)

        params = TextChatParameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        self.model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=project_id,
            params=params
        )

    def generate_response(self, encoded_image, prompt):
        try:
            logger.info("Sending request to LLM with prompt length: %d", len(prompt))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + encoded_image,
                            }
                        }
                    ]
                }
            ]

            response = self.model.chat(messages=messages)
            content = response['choices'][0]['message']['content']

            logger.info("Received response with length: %d", len(content))

            if len(content) >= 7900:
                logger.warning("Response may be truncated (length: %d)", len(content))

            return content

        except Exception as e:
            logger.error("Error generating response: %s", str(e))
            return f"Error generating response: {e}"

    def generate_fashion_response(
        self,
        user_image_base64,
        matched_row,
        all_items,
        similarity_score,
        threshold=0.8
    ):
        items_list = []
        for _, row in all_items.iterrows():
            item_name = str(row.get('Item Name', 'Fashion Item')).strip()
            raw_price = str(row.get('Price', 'N/A')).replace('$', '').strip()
            link = str(row.get('Link', '')).strip()

            search_query = urllib.parse.quote(f"{item_name} clothing buy online")
            search_fallback = f"https://www.google.com/search?q={search_query}"

            if link and link.startswith("http"):
                item_str = (
                    f"{item_name} (${raw_price}): "
                    f"[Direct Link]({link}) | [Search Item]({search_fallback})"
                )
            else:
                item_str = (
                    f"{item_name} (${raw_price}): "
                    f"[Search Item]({search_fallback})"
                )

            items_list.append(item_str)

        items_description = "\n".join([f"- {item}" for item in items_list])

        if similarity_score >= threshold:
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"ITEM DETAILS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Identify and describe the clothing items objectively (colors, patterns, materials)\n"
                "2. Categorize the overall style (business, casual, etc.)\n"
                "3. Include the ITEM DETAILS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )
        else:
            assistant_prompt = (
                f"You're conducting a professional retail catalog analysis. "
                f"This image shows standard clothing items available in department stores. "
                f"Focus exclusively on professional fashion analysis for a clothing retailer. "
                f"SIMILAR ITEMS (always include this section in your response):\n{items_description}\n\n"
                "Please:\n"
                "1. Note these are similar but not exact items\n"
                "2. Identify clothing elements objectively (colors, patterns, materials)\n" 
                "3. Include the SIMILAR ITEMS section at the end\n\n"
                "This is for a professional retail catalog. Use formal, clinical language."
            )

        response = self.generate_response(user_image_base64, assistant_prompt)

        if len(response) < 100:
            logger.info("Response appears incomplete, creating basic response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response = (
                f"# Fashion Analysis\n\n"
                f"This outfit features a collection of carefully coordinated pieces.\n\n"
                f"{section_header}\n{items_description}"
            )
        elif "ITEM DETAILS:" not in response and "SIMILAR ITEMS:" not in response:
            logger.info("Item details section missing from response")
            section_header = "ITEM DETAILS:" if similarity_score >= threshold else "SIMILAR ITEMS:"
            response += f"\n\n{section_header}\n{items_description}"

        return response
