"""
Test script to debug OpenAI inference with 3 samples.
Run with: python test_openai.py
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

def parse_input_for_openai(input_text, dataset):
    """
    Parse the processed input text to extract conversation_history, target_utterance, and audio_features.
    """
    # Extract the conversation between ### ### markers
    # Note: The text contains "'### ###'" as description, so we need to find the ACTUAL markers
    # The actual conversation starts after "speakers." or similar, followed by ### and Speaker_
    conv_match = re.search(r'speakers\.\s*###(.+?)###', input_text, re.DOTALL)
    if not conv_match:
        # Fallback: find ### followed by Speaker content
        conv_match = re.search(r'###\s*(\t?\s*Speaker_.+?)###', input_text, re.DOTALL)
    conversation_history = ""
    if conv_match:
        conversation_history = conv_match.group(1).strip()

    # Extract target utterance (from "For Speaker_X:..." pattern)
    target_match = re.search(r'For\s+(Speaker_\d+:["\'][^"\']+["\'])', input_text)
    target_utterance = ""
    if target_match:
        target_utterance = target_match.group(1).replace('<', '').replace('>', '')
    else:
        # Try to get the last utterance from conversation
        speaker_utterances = re.findall(r'Speaker_\d+:["\'][^"\']+["\']', conversation_history)
        if speaker_utterances:
            target_utterance = speaker_utterances[-1]

    # Extract audio features - try multiple patterns
    audio_features = ""

    # Pattern 1: "Target speech characteristics: ..."
    audio_match = re.search(r'Target speech characteristics:\s*([^.]+\.)', input_text, re.IGNORECASE)
    if audio_match:
        audio_features = audio_match.group(1).strip()

    # Pattern 2: "Audio description of target utterance: ..."
    if not audio_features:
        audio_match = re.search(r'Audio description of target utterance:\s*([^.]+\.)', input_text, re.IGNORECASE)
        if audio_match:
            audio_features = audio_match.group(1).strip()

    # Pattern 3: Check for features between ### and "For" or newline
    if not audio_features:
        after_conv_match = re.search(r'###\s*([^#]+?)(?:\n\n|For\s+Speaker)', input_text, re.DOTALL)
        if after_conv_match:
            potential_features = after_conv_match.group(1).strip()
            if any(kw in potential_features.lower() for kw in ['volume', 'pitch', 'speaking rate', 'variation']):
                audio_features = potential_features

    return conversation_history, target_utterance, audio_features


def test_parsing():
    """Test the parsing function with sample inputs."""

    test_inputs = [
        'Now you are expert of sentiment and emotional analysis.The following conversation noted between \'### ###\' involves several speakers.  ### \t Speaker_0:"Guess what?" ### Target speech characteristics: moderate volume with moderate variation, moderate pitch with very low variation, very low speaking rate.\n\nFor  Speaker_0:"Guess what?", based on the context and audio features, select one emotion from [happy, sad, neutral, angry, excited, frustrated]. Output your answer in the following JSON format:\n{"detected_emotion_label": "ONE_WORD_EMOTION_ONLY", "reason": "brief explanation"}\nIMPORTANT: The detected_emotion_label field must contain ONLY one emotion word from the list above, without any additional text or explanation. \nAnswer: ',

        'Now you are expert of sentiment and emotional analysis.The following conversation noted between \'### ###\' involves several speakers.  ### \t Speaker_0:"Guess what?"\t Speaker_1:"what?" ### Target speech characteristics: very low volume with very low variation, moderate pitch with very low variation, moderate speaking rate.\n\nFor  Speaker_1:"what?", based on the context and audio features, select one emotion from [happy, sad, neutral, angry, excited, frustrated]. Output your answer in the following JSON format:\n{"detected_emotion_label": "ONE_WORD_EMOTION_ONLY", "reason": "brief explanation"}\nIMPORTANT: The detected_emotion_label field must contain ONLY one emotion word from the list above, without any additional text or explanation. \nAnswer: ',

        'Now you are expert of sentiment and emotional analysis.The following conversation noted between \'### ###\' involves several speakers.  ### \t Speaker_0:"Guess what?"\t Speaker_1:"what?"\t Speaker_0:"I did it, I asked her to marry me." ### Target speech characteristics: moderate volume with moderate variation, high pitch with moderate variation, moderate speaking rate.\n\nFor  Speaker_0:"I did it, I asked her to marry me.", based on the context and audio features, select one emotion from [happy, sad, neutral, angry, excited, frustrated]. Output your answer in the following JSON format:\n{"detected_emotion_label": "ONE_WORD_EMOTION_ONLY", "reason": "brief explanation"}\nIMPORTANT: The detected_emotion_label field must contain ONLY one emotion word from the list above, without any additional text or explanation. \nAnswer: ',
    ]

    print("=" * 80)
    print("TESTING PARSING FUNCTION")
    print("=" * 80)

    for i, input_text in enumerate(test_inputs):
        print(f"\n--- Sample {i+1} ---")
        conversation, target, audio = parse_input_for_openai(input_text, 'iemocap')
        print(f"Conversation: {conversation}")
        print(f"Target utterance: {target}")
        print(f"Audio features: {audio}")
        print()

    return test_inputs


def test_openai_api(test_inputs):
    """Test the OpenAI API with parsed inputs."""

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        return
    print("checking error:")
    print(api_key)
    client = OpenAI(api_key=api_key)
    label_str = '"happy", "sad", "neutral", "angry", "excited", "frustrated"'

    print("=" * 80)
    print("TESTING OPENAI API")
    print("=" * 80)

    for i, input_text in enumerate(test_inputs):
        print(f"\n--- Sample {i+1} ---")

        # Parse the input
        conversation, target, audio = parse_input_for_openai(input_text, 'iemocap')

        # Create the prompt
        developer_message = f"""Now you are expert of emotional analysis for dialogues. Please select one emotion label word from [{label_str}] and always respond in strict json format."""

        user_message = f"""You will be analyzing a dialogue to identify the most dominant emotion expressed in a target utterance. You will need to consider both the conversation context and audio features to make your determination.

input:
Conversation history: {conversation}
Target utterance: {target}
Audio features: {audio if audio else "Not available"}

Your task is to select the single emotion that best represents the dominant emotion of the target utterance. You must choose from this list of emotions: [{label_str}]

Consider:
- What is being said in the target utterance and how it relates to the conversation context
- The audio features (pitch, volume, speaking rate) and what emotions they indicate
- How the conversation context influences the emotional state

Provide your answer in this JSON format:
{{
"emotion_label": "one emotion from the list",
"reasoning": "brief explanation referencing both context and audio features"
}}"""

        messages = [
            {"role": "developer", "content": developer_message},
            {"role": "user", "content": user_message}
        ]

        print(f"Sending to API:")
        print(f"  Conversation: {conversation[:100]}...")
        print(f"  Target: {target}")
        print(f"  Audio: {audio}")

        try:
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            '''
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                max_completion_tokens=200,
                response_format={"type": "json_object"}
            )
            '''
            output = response.choices[0].message.content
            print(f"\nAPI Response:\n{output}")
        except Exception as e:
            print(f"API Error: {e}")

        print()


if __name__ == "__main__":
    test_inputs = test_parsing()

    print("\n" + "=" * 80)
    response = input("Parsing looks correct? Run OpenAI API test? (y/n): ")
    if response.lower() == 'y':
        test_openai_api(test_inputs)
