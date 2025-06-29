from transformers import pipeline
import torch

def main():
    # Set device: 0 for GPU if available, else CPU (-1)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Device set to {'GPU' if device == 0 else 'CPU'}")

    # Load summarization pipeline with BART large CNN model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    # Get input story/paragraph from user
    story_text = input("Enter story/paragraph: ")

    print("\nGenerating Summary...\n")

    # Generate summary with controlled length
    summary = summarizer(
        story_text,
        max_length=90,
        min_length=40,
        do_sample=False,
        early_stopping=True
    )

    print("Generated Summary:\n")
    print(summary[0]['summary_text'])

if __name__ == "__main__":
    main()
