# Fashion Style Analyzer (Multimodal AI & Vector Search)

An end-to-end multimodal fashion recommendation application built with **PyTorch**, **IBM watsonx AI**, and **Gradio**.

## Features
- **Visual Embeddings**: Uses a pre-trained **ResNet-50** network to encode outfit photos into high-dimensional feature vectors.
- **Vector Similarity Search**: Finds nearest matches in a curated fashion catalog using cosine similarity.
- **Multimodal LLM Analysis**: Leverages **Llama Vision** via IBM watsonx AI to deliver formal fashion catalog critiques, garment identification, and style breakdowns.
- **Interactive UI**: Gradio web interface featuring image uploads, example presets, and markdown rendering with automatic search link fallbacks.

## Project Structure
```text
style-finder/
├── app.py                      # Main Gradio application entry point
├── config.py                   # Model IDs, thresholds, and environment settings
├── models/
│   ├── image_processor.py      # ResNet-50 vector encoding and cosine similarity search
│   └── llm_service.py          # IBM watsonx AI Llama Vision interface
├── utils/
│   └── helpers.py              # Catalog querying and markdown formatting helpers
├── examples/                   # Sample images for testing
├── swift-style-embeddings.pkl  # Precomputed catalog feature embeddings
└── requirements.txt            # Python dependencies
```

## Setup & Running Locally

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   python app.py
   ```
