"""
Test script to debug OpenAI VAD inference with sample data.
"""

import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def extract_vad_from_output(output):
    """Extract VAD values from model output."""
    if not output or output.strip() == '':
        return None, None, None

    # Strategy 1: Extract from <answer> tags
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        try:
            parsed = json.loads(answer_content)
            v = int(parsed.get('v_value', 3))
            a = int(parsed.get('a_value', 3))
            d = int(parsed.get('d_value', 3))
            return v, a, d
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 2: Find JSON object
    json_match = re.search(r'\{[^{}]*"v_value"[^{}]*\}', output)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            v = int(parsed.get('v_value', 3))
            a = int(parsed.get('a_value', 3))
            d = int(parsed.get('d_value', 3))
            return v, a, d
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Strategy 3: Regex patterns
    v_match = re.search(r'"?v_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    a_match = re.search(r'"?a_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)
    d_match = re.search(r'"?d_value"?\s*:\s*"?(\d)"?', output, re.IGNORECASE)

    if v_match and a_match and d_match:
        return int(v_match.group(1)), int(a_match.group(1)), int(d_match.group(1))

    return None, None, None


def create_vad_prompt(conversation_history, target_utterance, audio_features):
    """Create VAD evaluation prompt."""
    developer_message = """Now you are expert of evaluating emotions for dialogues based on Valence-Arousal-Dominance (VAD) Model. Please rate the emotion on a scale of 1-5 (only give integer) for each dimension."""

    user_message = f"""You will be analyzing a dialogue to evaluate the emotion expressed in a specific target utterance using the Valence-Arousal-Dominance (VAD) Model. The VAD model measures emotion across three dimensions:
* Valence: How positive or negative the emotion is (1 = very negative, 5 = very positive)
* Arousal: The intensity or energy level of the emotion (1 = very calm/passive, 5 = very excited/active)
* Dominance: The degree of control or power expressed (1 = very submissive/controlled, 5 = very dominant/in-control)

Input:
The conversation history leading up to the target utterance:
{conversation_history}

Here is the target utterance you need to analyze:
{target_utterance}

Here are the audio features (if available) for the last three utterances in the conversation:
{audio_features}

Your task is to rate the emotion expressed in the target utterance on a scale of 1-5 for each of the three VAD dimensions. You must provide integer values only (no decimals).

Provide your final ratings in valid JSON format inside <answer> tags. The JSON must have exactly this structure:
{{ "v_value": "your integer rating for valence", "a_value": "your integer rating for arousal", "d_value": "your integer rating for dominance" }}

Remember:
* All ratings must be integers from 1 to 5
* Consider both conversational context and audio features (when available)
* The ratings should reflect the emotion in the target utterance specifically, not the overall conversation"""

    return developer_message, user_message


def test_vad_samples():
    """Test with sample VAD data."""

    # Sample test cases with expected VAD ranges
    test_cases = [
        {
            "conversation_history": 'Speaker_F: "Excuse me."',
            "target_utterance": 'Speaker_M: "Do you have your forms?"',
            "audio_features": "moderate volume with low variation, moderate pitch, moderate speaking rate",
            "expected": {"valence": 2.5, "arousal": 2.0, "dominance": 2.5}  # Neutral/slightly frustrated
        },
        {
            "conversation_history": '''Speaker_F: "Is there a problem?"
Speaker_M: "Who told you to get in this line?"''',
            "target_utterance": 'Speaker_F: "You did. You were standing at the beginning and you directed me."',
            "audio_features": "moderate volume with moderate variation, moderate pitch with moderate variation, moderate speaking rate",
            "expected": {"valence": 2.5, "arousal": 3.0, "dominance": 2.5}  # Defensive
        },
        {
            "conversation_history": '''Speaker_F: "Did you get the mail? So you saw my letter?"
Speaker_M: "It's not fair."''',
            "target_utterance": 'Speaker_F: "Yeah. I know."',
            "audio_features": "low volume with low variation, low pitch with low variation, slow speaking rate",
            "expected": {"valence": 2.0, "arousal": 1.5, "dominance": 1.5}  # Sad/resigned
        },
        {
            "conversation_history": '''Speaker_0: "I have some news about the job."
Speaker_1: "What is it? Did you hear back?"''',
            "target_utterance": 'Speaker_0: "I got it! They offered me the position!"',
            "audio_features": "high volume with high variation, high pitch with high variation, fast speaking rate",
            "expected": {"valence": 5.0, "arousal": 4.0, "dominance": 4.0}  # Excited/happy
        },
    ]

    return test_cases


def run_test():
    """Run the VAD test."""

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment or .env file")
        return

    client = OpenAI(api_key=api_key)
    test_cases = test_vad_samples()

    print("=" * 80)
    print("TESTING VAD OpenAI API")
    print("=" * 80)

    for i, test in enumerate(test_cases):
        print(f"\n--- Test {i+1}: ---")
        print(f"Target: {test['target_utterance']}")
        print(f"Audio: {test['audio_features']}")
        print(f"Expected (approx): V={test['expected']['valence']}, A={test['expected']['arousal']}, D={test['expected']['dominance']}")

        developer_msg, user_msg = create_vad_prompt(
            test['conversation_history'],
            test['target_utterance'],
            test['audio_features']
        )

        messages = [
            {"role": "developer", "content": developer_msg},
            {"role": "user", "content": user_msg}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            output = response.choices[0].message.content

            # Extract VAD values
            v, a, d = extract_vad_from_output(output)

            print(f"\nAPI Response:")
            print(output[:500] + "..." if len(output) > 500 else output)
            print(f"\nExtracted: V={v}, A={a}, D={d}")

            if v and a and d:
                exp = test['expected']
                print(f"Difference from expected: V={abs(v - exp['valence']):.1f}, A={abs(a - exp['arousal']):.1f}, D={abs(d - exp['dominance']):.1f}")
            else:
                print("WARNING: Failed to extract VAD values!")

        except Exception as e:
            print(f"API Error: {e}")

        print("-" * 40)


if __name__ == "__main__":
    run_test()
