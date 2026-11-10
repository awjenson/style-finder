"""
Utility functions for the Style Finder application.
"""

import logging
import re
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_all_items_for_image(image_url: str, dataset: pd.DataFrame) -> pd.DataFrame:
    if dataset is None or dataset.empty or "Image URL" not in dataset.columns:
        logger.warning("Dataset is invalid or missing 'Image URL' column.")
        return pd.DataFrame()

    related_items = dataset[dataset["Image URL"] == image_url]
    logger.info("Found %d items related to image URL: %s", len(related_items), image_url)
    return related_items


def format_alternatives_response(
    user_response: str,
    alternatives: dict,
    similarity_score: float,
    threshold: float = 0.8
) -> str:
    rejection_phrases = [
        "I'm not able to provide",
        "I cannot",
        "I apologize, but",
        "I don't feel comfortable"
    ]

    if not user_response or any(phrase in user_response for phrase in rejection_phrases):
        user_response = "## Fashion Analysis Results\n\nHere are the items detected in your image:"
    
    header = (
        "\n\n## Similar Items Found\n\nHere are some similar items we found:\n"
        if similarity_score >= threshold
        else "\n\n## Similar Items Found\n\nHere are some visually similar items:\n"
    )
    enhanced_response = user_response + header
    
    items_added = 0
    max_items = 10
    
    if isinstance(alternatives, dict):
        for item, alts in alternatives.items():
            enhanced_response += f"\n### {item}:\n"
            if alts and isinstance(alts, list):
                for alt in alts[:3]:
                    if items_added >= max_items:
                        break
                    title = alt.get("title", "Item")
                    price = alt.get("price", "N/A")
                    source = alt.get("source", "Retailer")
                    link = alt.get("link", "#")
                    enhanced_response += f"- {title} for {price} from {source} ([Buy it here]({link}))\n"
                    items_added += 1
            else:
                enhanced_response += "- No alternatives found.\n"
    
    return enhanced_response


def process_response(response: str) -> str:
    if not response:
        logger.warning("Empty response received")
        return "# Fashion Analysis\n\nNo detailed analysis was generated. Please refer to the item details below."
    
    rejection_phrases = [
        "I'm not able to provide",
        "I cannot provide",
        "I apologize, but I cannot",
        "I don't feel comfortable",
        "violated our content policy"
    ]
    
    if any(phrase in response for phrase in rejection_phrases):
        logger.warning("Model rejected request; recovering item details section.")
        items_section = ""
        
        if "ITEM DETAILS:" in response:
            items_section = "## Item Details\n\n" + response.split("ITEM DETAILS:")[1].strip()
        elif "SIMILAR ITEMS:" in response:
            items_section = "## Similar Items\n\n" + response.split("SIMILAR ITEMS:")[1].strip()
        
        if items_section:
            response = "# Fashion Analysis\n\nHere are the items detected in your image:\n\n" + items_section
    
    processed = response.replace("$", r"\$")
    processed = processed.replace("ITEM DETAILS:", "## Item Details")
    processed = processed.replace("SIMILAR ITEMS:", "## Similar Items")
    
    if not processed.lstrip().startswith("#"):
        processed = "# Fashion Analysis\n\n" + processed
    
    processed = re.sub(r"^\*\s+", "- ", processed, flags=re.MULTILINE)
    return processed
