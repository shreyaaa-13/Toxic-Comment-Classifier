"""
Generate sample dataset for testing the Toxic Comment Classifier
This creates a synthetic dataset when the Kaggle dataset is not available
"""

import pandas as pd
import numpy as np
import os

# Sample toxic and non-toxic comments
TOXIC_COMMENTS = [
    "You are an idiot and nobody likes you",
    "This is the worst thing I've ever seen, you're terrible",
    "Shut up, you don't know what you're talking about",
    "I hate this so much, it's complete garbage",
    "You're a loser and always will be",
    "This is stupid and so are you",
    "Get lost, nobody wants you here",
    "You're pathetic and worthless",
    "I can't stand people like you",
    "This is trash and you should be ashamed",
    "You're a complete moron",
    "Nobody cares about your stupid opinion",
    "You're the worst person ever",
    "This is absolutely terrible and you suck",
    "I hope you fail at everything you do",
    "You're disgusting and horrible",
    "Shut your mouth, you fool",
    "This is garbage and you're an idiot",
    "You're a waste of space",
    "I hate everything about this and you",
    "You're so dumb it hurts",
    "This is the stupidest thing ever",
    "You're a terrible person",
    "Nobody likes you or your ideas",
    "You're completely useless",
    "This is awful and you should feel bad",
    "You're an embarrassment",
    "I can't believe how stupid this is",
    "You're a joke and everyone knows it",
    "This is horrible and so are you",
]

NON_TOXIC_COMMENTS = [
    "This is a great article, thanks for sharing!",
    "I really appreciate your perspective on this topic",
    "Interesting point of view, I hadn't considered that",
    "Thank you for the helpful information",
    "I agree with your analysis, well done",
    "This is very informative and well-written",
    "Great work, keep it up!",
    "I learned something new today, thanks!",
    "This is exactly what I was looking for",
    "Excellent explanation, very clear",
    "I love this community, everyone is so helpful",
    "Thanks for taking the time to explain this",
    "This is a wonderful resource",
    "I appreciate your effort in creating this",
    "Very insightful, thank you for sharing",
    "This helped me understand the concept better",
    "Great job on this project!",
    "I'm glad I found this information",
    "This is really useful, thanks!",
    "Well explained and easy to follow",
    "I enjoyed reading this article",
    "This is a fantastic resource for beginners",
    "Thank you for your contribution",
    "I found this very helpful",
    "Excellent work, very professional",
    "This is exactly what I needed",
    "Great content, keep posting!",
    "I appreciate the detailed explanation",
    "This is very well organized",
    "Thank you for sharing your knowledge",
    "This is a valuable resource",
    "I'm impressed by the quality of this",
    "Very helpful tutorial, thanks!",
    "This is clear and concise",
    "I learned a lot from this",
    "Great information, well presented",
    "This is very educational",
    "Thank you for the clear explanation",
    "I found this very interesting",
    "Excellent resource, highly recommended",
]


def generate_sample_dataset(n_samples=2000, output_path='data/train.csv', random_state=42):
    """
    Generate a sample dataset for testing
    
    Args:
        n_samples (int): Total number of samples to generate
        output_path (str): Path to save the CSV file
        random_state (int): Random seed for reproducibility
    """
    np.random.seed(random_state)
    
    # Calculate number of toxic and non-toxic samples (40% toxic, 60% non-toxic)
    n_toxic = int(n_samples * 0.4)
    n_non_toxic = n_samples - n_toxic
    
    # Generate samples by repeating and adding variations
    toxic_samples = []
    non_toxic_samples = []
    
    # Generate toxic comments
    for i in range(n_toxic):
        base_comment = np.random.choice(TOXIC_COMMENTS)
        # Add some variation
        if np.random.random() > 0.5:
            base_comment = base_comment + " " + np.random.choice(["!!!", ".", "..."])
        toxic_samples.append(base_comment)
    
    # Generate non-toxic comments
    for i in range(n_non_toxic):
        base_comment = np.random.choice(NON_TOXIC_COMMENTS)
        # Add some variation
        if np.random.random() > 0.5:
            base_comment = base_comment + " " + np.random.choice(["!", ".", ":)"])
        non_toxic_samples.append(base_comment)
    
    # Create dataframe
    data = {
        'id': range(n_samples),
        'comment_text': toxic_samples + non_toxic_samples,
        'toxic': [1] * n_toxic + [0] * n_non_toxic,
        'severe_toxic': [np.random.randint(0, 2) if i < n_toxic else 0 for i in range(n_samples)],
        'obscene': [np.random.randint(0, 2) if i < n_toxic else 0 for i in range(n_samples)],
        'threat': [np.random.randint(0, 2) if i < n_toxic else 0 for i in range(n_samples)],
        'insult': [np.random.randint(0, 2) if i < n_toxic else 0 for i in range(n_samples)],
        'identity_hate': [np.random.randint(0, 2) if i < n_toxic else 0 for i in range(n_samples)],
    }
    
    df = pd.DataFrame(data)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated {n_samples} samples")
    print(f"   - Toxic: {n_toxic} ({n_toxic/n_samples*100:.1f}%)")
    print(f"   - Non-toxic: {n_non_toxic} ({n_non_toxic/n_samples*100:.1f}%)")
    print(f"   - Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample dataset
    df = generate_sample_dataset(n_samples=2000, output_path='data/train.csv')
    
    # Display sample
    print("\n📊 Sample data:")
    print(df.head(10))
    
    print("\n📈 Dataset statistics:")
    print(df.describe())
