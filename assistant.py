# Project 7 — LLM Assistant
# Uses the Anthropic API to interact with Claude
# and perform text analysis tasks.

import os
from dotenv import load_dotenv
import anthropic


def load_client():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    return client


def ask_claude(client, prompt):
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text


def analyze_sentiment(client, text):
    prompt = f"Analyze the sentiment of the following text in one sentence: {text}"
    return ask_claude(client, prompt)


def summarize_text(client, text):
    prompt = f"Summarize the following text in one sentence: {text}"
    return ask_claude(client, prompt)


def extract_keywords(client, text):
    prompt = f"Extract the 5 most important keywords from the following text, return them as a comma-separated list: {text}"
    return ask_claude(client, prompt)


def main():
    client = load_client()
    # Sample text for demonstration
    text = """
    Ridley Scott's Gladiator is a film that operates on pure, unapologetic spectacle — 
    and somehow, against all odds, it earns every moment of it. 
    This is not subtle cinema. It does not whisper. It roars.

    Russell Crowe plays Maximus, a Roman general betrayed by a jealous emperor's son,
    stripped of everything he loves, and thrown into the arena as a slave.
    It is a simple story, almost mythological in its bones, and Crowe carries it with a
    brooding, physical intensity that makes you forget how familiar the premise actually is.
    He does not play a hero. He plays a man with nothing left to lose,
    which is far more dangerous and far more interesting.

    Joaquin Phoenix, as the poisonous Emperor Commodus, delivers one of the most underrated
    villain performances of his generation. He is not monstrous in the obvious way.
    He is petty, wounded, and desperate for love — which makes him infinitely more unsettling
    than any straightforward tyrant. Every scene between him and Crowe crackles with tension.

    The battle sequences are brutal and kinetic, edited with a restless energy that puts you
    inside the chaos rather than above it. Rome itself feels enormous and indifferent,
    a civilization that has confused bloodshed with culture.

    Where the film stumbles is in its quieter moments. Some of the dialogue strains under
    the weight of its own grandeur, reaching for profundity and landing instead on cliché.
    The romantic and spiritual subplots feel thin against the film's overwhelming appetite for scale.

    And yet. When the final act arrives, when the dust settles and the crowd falls silent,
    Gladiator delivers something genuinely moving. It is a film about dignity — about whether
    a man can hold onto his humanity when the world is designed to strip it away.

    The answer, Scott suggests, is yes. But the cost is everything."""
    
    print("=== Sentiment ===")
    print(analyze_sentiment(client, text))
    
    print("=== Summary ===")
    print(summarize_text(client, text))
    
    print("=== Keywords ===")
    print(extract_keywords(client, text))


if __name__ == "__main__":
    main()