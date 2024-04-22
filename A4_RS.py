import torch
import random
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from rouge import Rouge
import sys

# Define the device for processing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load fine-tuned GPT-2 tokenizer and model
try:
    tokenizer = GPT2Tokenizer.from_pretrained("ft_model")
    model = GPT2LMHeadModel.from_pretrained("ft_model").to(device)
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

# Define function for model inference
def topk(probs, n=9):
    # The scores are initially softmaxed to convert to probabilities
    probs = torch.softmax(probs, dim=-1)
    
    # PyTorch has its own topk method, which we use here
    tokensProb, topIx = torch.topk(probs, k=n)
    
    # The new selection pool (9 choices) is normalized
    tokensProb = tokensProb / torch.sum(tokensProb)

    # Send to CPU for numpy handling
    tokensProb = tokensProb.cpu().detach().numpy()

    # Make a random choice from the pool based on the new prob distribution
    choice = np.random.choice(n, 1, p=tokensProb)
    tokenId = topIx[choice][0]

    return int(tokenId)

def model_infer(model, tokenizer, review, max_length=15):
    # Preprocess the init token (task designator)
    review_encoded = tokenizer.encode(review)
    result = review_encoded
    initial_input = torch.tensor(review_encoded).unsqueeze(0).to(device)

    with torch.set_grad_enabled(False):
        # Feed the init token to the model
        output = model(initial_input)

        # Flatten the logits at the final time step
        logits = output.logits[0, -1]

        # Make a top-k choice and append to the result
        result.append(topk(logits))

        # For max_length times:
        for _ in range(max_length):
            # Feed the current sequence to the model and make a choice
            input = torch.tensor(result).unsqueeze(0).to(device)
            output = model(input)
            logits = output.logits[0, -1]
            res_id = topk(logits)

            # If the chosen token is EOS, return the result
            if res_id == tokenizer.eos_token_id:
                return tokenizer.decode(result)
            else:  # Append to the sequence 
                result.append(res_id)
    # If no EOS is generated, return after the max_len
    return tokenizer.decode(result)

# Compute ROUGE scores
def compute_rouge_scores(actual_summary, generated_summary):
    rouge = Rouge()
    scores = rouge.get_scores(actual_summary, generated_summary, avg=True)
    return scores

# Take user input and generate summaries
def generate_and_evaluate_summary():
    review_text = input("Enter the review text: ").strip()
    actual_summary = input("Enter the actual summary: ").strip()

    # Validate user input
    if not review_text or not actual_summary:
        print("Error: Please provide non-empty review text and summary.")
        return

    try:
        # Generate summary
        generated_summary = model_infer(model, tokenizer, review_text)

        # Compute ROUGE scores
        rouge_scores = compute_rouge_scores(actual_summary, generated_summary)

        # Print results
        print("\nGenerated Summary:", generated_summary)
        print("ROUGE-1: Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
            rouge_scores["rouge-1"]["p"], rouge_scores["rouge-1"]["r"], rouge_scores["rouge-1"]["f"]))
        print("ROUGE-2: Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
            rouge_scores["rouge-2"]["p"], rouge_scores["rouge-2"]["r"], rouge_scores["rouge-2"]["f"]))
        print("ROUGE-L: Precision: {:.2f}, Recall: {:.2f}, F1-Score: {:.2f}".format(
            rouge_scores["rouge-l"]["p"], rouge_scores["rouge-l"]["r"], rouge_scores["rouge-l"]["f"]))
    except Exception as e:
        print("Error during summary generation:", e)

# Execute the main function
if __name__ == "__main__":
    generate_and_evaluate_summary()
