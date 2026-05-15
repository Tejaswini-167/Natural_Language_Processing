# ============================================================
# Lab Sheet 9: Building HMM for Title Case Conversion
# ============================================================

# Aim:
# To design and implement a Hidden Markov Model (HMM) to convert
# noisy or lowercase text into proper title case by modeling it
# as a sequence labeling problem.

# ============================================================
# Import Required Library
# ============================================================

from collections import defaultdict


# ============================================================
# Define Hidden States
# ============================================================

states = ["LOWER", "TITLE"]


# ============================================================
# Training Data
# ============================================================

training_data = [
    ("this", "LOWER"),
    ("is", "LOWER"),
    ("nasa", "LOWER"),
    ("research", "LOWER"),

    ("This", "TITLE"),
    ("Is", "TITLE"),
    ("NASA", "TITLE"),
    ("Research", "TITLE")
]


# ============================================================
# Special Words That Should Stay Fully Uppercase
# ============================================================

special_upper = {"nasa"}


# ============================================================
# Count Emission Frequencies
# ============================================================

emission_counts = defaultdict(lambda: defaultdict(int))
state_counts = defaultdict(int)

for word, state in training_data:
    state_counts[state] += 1
    emission_counts[state][word.lower()] += 1


# ============================================================
# Convert Emission Counts to Probabilities
# ============================================================

emission_prob = {
    state: {
        word: count / state_counts[state]
        for word, count in emission_counts[state].items()
    }
    for state in states
}


# ============================================================
# Define Transition Probabilities
# ============================================================

transition_prob = {
    "LOWER": {"LOWER": 0.5, "TITLE": 0.5},
    "TITLE": {"LOWER": 0.5, "TITLE": 0.5}
}


# ============================================================
# Viterbi Algorithm
# ============================================================

def viterbi(words):
    V = [{}]
    path = {}

    # Initialization step
    for state in states:
        V[0][state] = emission_prob[state].get(words[0], 1e-6)
        path[state] = [state]

    # Recursion step
    for i in range(1, len(words)):
        V.append({})
        new_path = {}

        for current_state in states:
            probability, previous_state = max(
                (
                    V[i - 1][prev_state]
                    * transition_prob[prev_state][current_state]
                    * emission_prob[current_state].get(words[i], 1e-6),
                    prev_state
                )
                for prev_state in states
            )

            V[i][current_state] = probability
            new_path[current_state] = path[previous_state] + [current_state]

        path = new_path

    # Termination step
    best_final_state = max(V[-1], key=V[-1].get)

    return path[best_final_state]


# ============================================================
# Title Case Conversion Function
# ============================================================

def convert_to_title_case(text):
    words = text.split()
    tags = viterbi([word.lower() for word in words])

    result = []

    for index, (word, tag) in enumerate(zip(words, tags)):

        # Rule 1: First word should always be title case
        if index == 0:
            result.append(word.capitalize())

        # Rule 2: Special words like NASA should remain uppercase
        elif word.lower() in special_upper:
            result.append(word.upper())

        # Rule 3: Apply HMM predicted tag
        elif tag == "TITLE":
            result.append(word.capitalize())

        else:
            result.append(word.lower())

    return " ".join(result)


# ============================================================
# Test the Model
# ============================================================

text = "this is nasa research"

print("Input :", text)
print("Output:", convert_to_title_case(text))