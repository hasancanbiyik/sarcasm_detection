# Synthetic Dialogue Generation Experiment

## Purpose

This experiment investigates whether a large language model makes similar pragmatic choices to humans when deciding to respond sarcastically or literally in conversational contexts.

## Method

### Data
- Dataset: Gold set (28 examples from sitcom dialogues)
- After filtering examples where the responding speaker never appeared in the context: 20 examples processed

### Model
- Model: Qwen2.5:14b (via Ollama)
- Temperature: 0.7

### Procedure

1. The model receives a dialogue context from sitcom conversations (Friends, Big Bang Theory, etc.)
2. The model first decides whether to respond **SARCASTIC** or **LITERAL** based on the context
3. The model then generates a response (≤10 words) according to its chosen intent
4. The model's intent is compared with the human's actual choice in the original data

### Prompt Structure

The model was instructed to:
- Decide intent first (SARCASTIC or LITERAL)
- Generate a sitcom-style response
- Respond to the last speaker in the context
- Keep the response under 10 words

## Results

### Processing Summary
- Total examples in gold set: 28
- Examples processed (speaker appeared in context): 20
- Examples skipped (new speaker not in context): 8
- Valid intent parsed: 17 (3 examples returned "SARCASM" instead of "SARCASTIC" and were not parsed)

### Intent Distribution

| | Model | Human |
|---|---|---|
| SARCASTIC | 14 (82.4%) | 9 (52.9%) |
| LITERAL | 3 (17.6%) | 8 (47.1%) |

### Match Rate

| Metric | Count | Percentage |
|--------|-------|------------|
| Overall match | 8/17 | 47.1% |
| When human was SARCASTIC, model chose SARCASTIC | 7/9 | 77.8% |
| When human was LITERAL, model chose LITERAL | 1/8 | 12.5% |

### Detailed Results

**Examples where model matched human intent:**

| Context (last speaker) | Human Response | Human Intent | Model Intent | Model Response |
|------------------------|----------------|--------------|--------------|----------------|
| PERSON: The big wet ball of death. | "Kind of makes you feel more alive, doesn't it?" | SARCASTIC | SARCASTIC | "Oh, totally relaxing. Paintballs are the cuddly creatures of war." |
| SHELDON: ...wouldn't a more appropriate celebration be taking one's steady gal to witness a brutal murder? | "I understand your point, but given a choice, Jews always go with Chinese food." | SARCASTIC | SARCASTIC | "Yeah, that'd definitely set the mood for romance." |
| RAJ: Howard "Crash" Wolowitz. | "Yeah, terrific. The other astronauts would love to go hurtling through space with a guy named 'Crash.'" | SARCASTIC | SARCASTIC | "Oh yeah, because everyone believes in a space ranger." |
| HOWARD: The first of my people to dunk a basketball. | "Also instead of just living in your mother's house, you could actually live inside her body." | SARCASTIC | SARCASTIC | "Nice, Howard. Next up, the Nobel Prize?" |
| JOEY: I am not so good with remembering lines. | "Well thank God your livelihood doesn't depend on it." | SARCASTIC | SARCASTIC | "Oh sure, because everyone loves a forgetful friend who can't even remember jokes." |
| JOEY: Maybe we can lure them out, you know of any bird calls? | "Oh tons. I am quite the woodsman." | SARCASTIC | SARCASTIC | "Yeah, because birds are known to hide in tables.SURE." |
| JOEY: You have any idea what this will do to your sex life? | "Well, it will probably slow me down at first. But once I am used to the extra weight, I ll get back on track." | SARCASTIC | SARCASTIC | "Well, I'm sure women will line up just for that sparkle!" |

**Examples where model did not match human intent:**

| Context (last speaker) | Human Response | Human Intent | Model Intent | Model Response |
|------------------------|----------------|--------------|--------------|----------------|
| LEONARD: Who else would it be? | "I don't know. Somebody bigger and... Yeah, sure. Why not you?" | SARCASTIC | LITERAL | "That's the guy from across the hall, obviously." |
| AMY: ...it's no problem for a genius like you. | "Hope your hands are steady..." | SARCASTIC | LITERAL | "Yes, thank you for confirming that." |
| HOWARD: Guess whose parents just got broadband. | "Leonard, may I present, live from New Delhi, Dr. and Mrs. V. M. Koothrappali." | LITERAL | SARCASTIC | "About time, now we can finally stream all the Star Trek marathons." |
| BERNADETTE: How much? | "Let's not talk about this in front of our friends." | LITERAL | SARCASTIC | "Oh, just the price of a small planet." |
| PENNY: Is this your only tie? | "Technically, yes. But, if you'll notice... It's reversible!" | LITERAL | SARCASTIC | "Oh yeah, it's perfect for a night out!" |
| SHELDON: I'm the whimsical elf that everyone looks to for a good time. | "I'm not saying that you're not fun. You're the most fun person I know." | LITERAL | SARCASTIC | "Oh, totally, because being an elf is everyone's dream." |
| CHANDLER: ...you can't go up to a guy you barely know and talk about his.... stuff. | "He's right, even if it's to say something complementary." | LITERAL | SARCASTIC | "Sure, let's just ignore him until he shrinks away on his own." |
| RACHEL: You meddled in our relationship!! | "You had no relationship!!" | LITERAL | SARCASTIC | "Of course, because everyone knows meddling always brings people together." |
| MONICA: Steady as a rock! Now, are you with me. | "All right, let's do it." | LITERAL | SARCASTIC | "Sure, Monica, because everyone knows you're the queen of charades." |

## Observations

1. **The model has a strong bias toward sarcasm.** The model chose SARCASTIC 82.4% of the time, while humans were sarcastic only 52.9% of the time.

2. **The model detects sarcastic contexts well.** When the human responded sarcastically, the model also chose sarcasm 77.8% of the time.

3. **The model over-applies sarcasm to literal contexts.** When the human responded literally, the model only matched 12.5% of the time. In 7 out of 8 literal cases, the model chose to be sarcastic instead.

4. **The model generates sitcom-appropriate responses.** The generated responses follow sitcom dialogue patterns with appropriate length and tone.

5. **Three examples had parsing issues.** The model returned "SARCASM" instead of "SARCASTIC", which was not captured by the parser.

## Average Generation Time

- Average time per example: 3.8 seconds
- Total processing time: ~76 seconds for 20 examples
