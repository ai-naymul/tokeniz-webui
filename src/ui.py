# app.py

import gradio as gr
from tokenizer_logic import create_tokenizer, test_tokenizer

def main():
    # Create Gradio interface for creating a tokenizer
    iface_create = gr.Interface(
        fn=create_tokenizer,
        inputs=[
            gr.Textbox(label="HuggingFace Dataset Name", placeholder="Enter the huggingface dataset name here..."),
            gr.Textbox(label="HuggingFace Dataset Subset", placeholder="Optional"),
            gr.Radio(label="Choose Tokenizer Type:", choices=["Byte-Pair Encoding", "Unigram", "WordPiece"]),
            gr.Slider(minimum=1, maximum=300000, label="Vocabulary Size:"),
            gr.Textbox(label="Special Tokens (comma-separated)", placeholder="[UNK],[CLS]"),
            gr.Slider(minimum=1, label="Minimum Token Frequency:"),
        ],
        outputs="text",
        title="Custom Tokenizer Creator",
        description="Create your own tokenizer by configuring options below."
    )
    
    # Create Gradio interface for testing the tokenizer
    iface_test = gr.Interface(
        fn=test_tokenizer,
        inputs=gr.Textbox(label="Input Text for Testing", placeholder="Type text to tokenize..."),
        outputs=["json"],
        title="Test Your Tokenizer",
        description="Use your trained tokenizer to encode text."
    )

    # Launch both interfaces in a tabbed format
    gr.TabbedInterface([iface_create, iface_test], ["Create Tokenizer", "Test Tokenizer"]).launch()

if __name__ == "__main__":
    main()