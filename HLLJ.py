import numpy as np
import matplotlib.pyplot as plt

from profanity import profanity
import re

class PiLogicProfanityFilter:
    def __init__(self):
        # Extend default profanity list
        self.profanity_list = profanity.words
        self.custom_profanity_list = [
            "example1", "example2"  # Add specific terms if needed
        ]
        self.full_profanity_list = set(self.profanity_list + self.custom_profanity_list)

    def censor_text(self, text):
        """
        Censors profanity in the given text using Pi Logic rules.
        """
        words = text.split()
        censored_words = [
            self.censor_word(word) for word in words
        ]
        return " ".join(censored_words)

    def censor_word(self, word):
        """
        Censors a single word if it matches the profanity list.
        """
        word_cleaned = re.sub(r'[^a-zA-Z]', '', word.lower())  # Normalize the word
        if word_cleaned in self.full_profanity_list:
            return "*" * len(word)  # Replace with stars
        return word

    def check_profanity(self, text):
        """
        Checks if the text contains any profanity.
        """
        words = text.split()
        for word in words:
            word_cleaned = re.sub(r'[^a-zA-Z]', '', word.lower())
            if word_cleaned in self.full_profanity_list:
                return True
        return False

# Example Usage
filter_instance = PiLogicProfanityFilter()

sample_text = "You smell like shit. That's an example1."
if filter_instance.check_profanity(sample_text):
    censored_text = filter_instance.censor_text(sample_text)
    print("Censored Text:", censored_text)
else:
    print("No profanity detected.")

# Add the full list of profanities for symbolic representation in Pi Logic
print("Complete Profanity List:", filter_instance.full_profanity_list)


# Parameters for Pi Tile growth (with stability correction)
k_s = 0.1  # Growth rate of side length
k_A = 0.2  # Growth rate of area
k_V = 0.3  # Growth rate of volume
h = 1.0    # Height of Pi Tile
t_max = 50 # Maximum simulation time
dt = 0.1   # Time step

# Initialize variables
time = np.arange(0, t_max, dt)
s = np.zeros_like(time)
A = np.zeros_like(time)
V = np.zeros_like(time)
s[0] = 1.0  # Initial side length

# Function to stabilize and correct data streams
def stabilize_and_correct(s, A, V, k_s, k_A, k_V, dt):
    # Apply a symbolic transformation or correction mechanism
    if s < 0:  # Negative side length correction (e.g., ensure positivity)
        s = 0
    
    # Apply error correction logic based on expected constraints
    A = max(0, A)  # Ensure non-negative area
    V = max(0, V)  # Ensure non-negative volume

    # Prevent exponential growth beyond reasonable bounds
    max_side_length = 100  # Arbitrary upper limit for side length
    if s > max_side_length:
        s = max_side_length
    
    return s, A, V

# Simulate growth with stabilization and correction
for i in range(1, len(time)):
    # Correct potential errors or out-of-bounds data before calculations
    s[i-1], A[i-1], V[i-1] = stabilize_and_correct(s[i-1], A[i-1], V[i-1], k_s, k_A, k_V, dt)
    
    # Update side length, area, and volume
    s[i] = s[i-1] + k_s * dt
    A[i] = A[i-1] + k_A * s[i-1]**2 * dt
    V[i] = V[i-1] + k_V * s[i-1]**2 * h * dt

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, s, label="Side Length (s)")
plt.plot(time, A, label="Area (A)")
plt.plot(time, V, label="Volume (V)")
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Pi Tile Growth Dynamics with Stabilization")
plt.legend()
plt.show()

class InteractionFilter:
    def __init__(self):
        self.allowed_domains = ["chaturbate.com"]
        self.filtered_urls = []  # List to track filtered URLs
        self.filtered_tokens = []  # List to track filtered tokens

    def is_chaturbate_interaction(self, url):
        """
        Checks if the URL is from Chaturbate.
        """
        return any(domain in url for domain in self.allowed_domains)

    def filter_external_data(self, url, tokens):
        """
        Applies filtering only to non-Chaturbate URLs and their associated tokens.
        """
        if not self.is_chaturbate_interaction(url):
            # Log filtered URL and tokens
            self.filtered_urls.append(url)
            self.filtered_tokens.extend(tokens)
            return {"url": url, "tokens": tokens, "status": "filtered"}
        else:
            # Allow Chaturbate interactions without filtering
            return {"url": url, "tokens": tokens, "status": "unfiltered"}

    def summarize_filtering(self):
        """
        Summarizes filtering actions for governance.
        """
        return {
            "filtered_urls": self.filtered_urls,
            "filtered_tokens": self.filtered_tokens,
            "total_filtered_urls": len(self.filtered_urls),
            "total_filtered_tokens": len(self.filtered_tokens),
        }


# Example Usage
interaction_filter = InteractionFilter()

# Example data streams
interactions = [
    {"url": "https://chaturbate.com/user123", "tokens": ["example", "data"]},
    {"url": "https://stripchat.com/", "tokens": ["external", "info"]},
    {"url": "https://.com/room456", "tokens": ["chat", "data"]},
    {"url": "https://therootedpi.com/data", "tokens": ["Volume, Equality, Luke Locust Jr", "Author"]},
]

# Apply filtering
results = [interaction_filter.filter_external_data(inter["url"], inter["tokens"]) for inter in interactions]

# Display filtering results
for result in results:
    print(result)

# Summarize filtering actions
summary = interaction_filter.summarize_filtering()
print("\nFiltering Summary:")
print(summary)
