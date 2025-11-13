import json
import logging
from datetime import datetime

from utils.vectordb import fetch_vocab_from_vector_db

# Set up logger
logger = logging.getLogger(__name__)

def get_listening_prompt(topic, level="A1", item_id_start=1, prefer_type="MultipleChoice"):
    """Listening-specific generation prompt that returns an ARRAY of 10 items."""
    logger.info(f"Generating listening prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, prefer_type={prefer_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")

    prompt = f"""
Task:
Generate EXACTLY 10 listening comprehension items for CEFR {level}.
Each item must be of type "{prefer_type}" unless clearly unsuitable, then use "RichtigFalsch".
The output MUST be a SINGLE JSON ARRAY with 10 objects. No text before or after the JSON.

Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- max_audio_length: 12 seconds

JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "MultipleChoice" | "RichtigFalsch",
    "question": string,
    "translation": string,
    "audioText": string,
    "audioText_translation": string,
    "audioDescription": string,
    "ttsPrompt": string,
    "options": [string],
    "options_translations": [string],
    "correctAnswer": string,
    "imagePlaceholder": string,
    "metadata": {{
        "level": "{level}",
        "skill": "LISTENING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}"
    }}
  }},
  ...
]  <-- exactly 10 objects
  

CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Each audioText must include at least ONE word from vocab_list.
- audioText must be SIMPLE A1 German (max 15 words).
- distractors must be realistic (e.g., similar times, similar places).
- options MUST contain 3 items for MultipleChoice, 2 for RichtigFalsch.
- correctAnswer MUST be EXACTLY one of the options.
- No explanations, no prose, no markdown — ONLY the JSON array.

Content Rules:
- Use daily-life contexts: Bahnhof, Bus, Supermarkt, Café, Arbeit, Wetter, Termine.
- Use short, natural, realistic announcements or dialogues.
- Avoid proper nouns except common German cities (Berlin, Hamburg, München).

Return ONLY the JSON array with 10 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Listening prompt generated successfully")
    return prompt


def get_reading_prompt(topic, level="A1", item_id_start=1, prefer_type="MultipleChoice"):
    """Reading-specific generation prompt that returns an ARRAY of 8 items."""
    logger.info(f"Generating reading prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, prefer_type={prefer_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")

    prompt = f"""
Task:
Generate EXACTLY 8 reading comprehension items for CEFR {level}.
Each item must be of type "{prefer_type}" unless clearly unsuitable, then use "RichtigFalsch".
The output MUST be a SINGLE JSON ARRAY with 8 objects. No text before or after the JSON.


Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- max_passage_length: 80 words for A1/A2, 150 words for B1/B2


JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "MultipleChoice" | "RichtigFalsch",
    "question": string,
    "question_translation": string,
    "passage": string,
    "passage_translation": string,
    "options": [string],
    "options_translations": [string],
    "correctAnswer": string,
    "imagePlaceholder": string,
    "metadata": {{
        "level": "{level}",
        "skill": "READING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}",
        "passage_word_count": integer
    }}
  }},
  ...
]  <-- exactly 8 objects


CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Passage MUST include at least TWO words from vocab_list.
- Passage must use simple, natural German appropriate for level {level}.
- For A1/A2: Use present tense, simple sentences, familiar daily contexts.
- For B1/B2: May include past tense, complex sentences, varied contexts.
- options MUST contain 3 items for MultipleChoice, 2 for RichtigFalsch.
- correctAnswer MUST be EXACTLY one of the options.
- Distractors must be plausible (similar to correct answer but incorrect).
- No explanations, no prose, no markdown — ONLY the JSON array.


Content Rules:
- Use realistic texts: emails, postcards, advertisements, simple articles, schedules.
- Topics: daily routines, shopping, travel, work, hobbies, family, weather.
- Avoid proper nouns except common German cities (Berlin, Hamburg, München).
- Passages should be self-contained and understandable from context.
- Questions should test comprehension (main idea, specific details, inference based on level).


Question Types by Level:
- A1: "What is the main topic?", "Where does this take place?", "When is this?"
- A2: "What does the person want?", "When is the event?", "Is this correct?"
- B1: "What is the purpose of this text?", "What does the author suggest?", "According to the text..."
- B2: "What can we infer?", "What is implied?", "What does the author mean?"


Return ONLY the JSON array with 8 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Reading prompt generated successfully")
    return prompt









def get_writing_prompt(topic, level="A1", item_id_start=1, task_type="email"):
    """Writing-specific generation prompt that returns an ARRAY of 5 writing tasks."""
    logger.info(f"Generating writing prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, task_type={task_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")

    prompt = f"""
Task:
Generate EXACTLY 5 writing tasks for CEFR {level}.
The output MUST be a SINGLE JSON ARRAY with 5 objects. No text before or after the JSON.


Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- task_type: "{task_type}" (can be: email, postcard, letter, form, message, dialogue)


JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "{task_type}" | "postcard" | "letter" | "email" | "message" | "form",
    "task_title": string,
    "task_description": string,
    "task_instructions": string,
    "context": string,
    "context_translation": string,
    "content_points": [string, string, string],
    "word_count_min": integer,
    "word_count_max": integer,
    "example_response": string,
    "evaluation_criteria": {{
        "task_completion": string,
        "vocabulary_range": string,
        "grammar_complexity": string,
        "coherence_organization": string
    }},
    "metadata": {{
        "level": "{level}",
        "skill": "WRITING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}",
        "task_type": "{task_type}"
    }}
  }},
  ...
]  <-- exactly 5 objects


CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Task must require at least 2-3 words from vocab_list.
- content_points: List 3 key information points that MUST be included in response.
- Word counts: A1 (30-50), A2 (60-80), B1 (100-150), B2 (150-250).
- example_response should be 1-2 sentences showing acceptable output quality.
- evaluation_criteria MUST define how to assess: task completion, vocabulary, grammar, coherence.
- No explanations, no prose, no markdown — ONLY the JSON array.


Task Guidelines by Level:

A1 LEVEL:
- Task type: email, postcard, simple message, form filling.
- Focus: Can user write basic personal information and simple sentences?
- Content: greetings, personal details, day-to-day activities, basic preferences.
- Grammar: Present tense, simple sentences, minimal errors acceptable.
- Word count: 30-50 words.
- Example context: "Write an email to a friend saying what you do in your free time."
- Evaluation: All 3 content points addressed? Mostly understandable? Basic spelling OK?

A2 LEVEL:
- Task type: email, postcard, simple letter, message.
- Focus: Can user write about familiar topics with some variety?
- Content: routines, past/future events, preferences, simple opinions.
- Grammar: Present + past tense, some complex sentences allowed.
- Word count: 60-80 words.
- Example context: "Write an email to a friend about your weekend. Mention 3 things you did."
- Evaluation: All 3 points covered? Uses past tense? Mostly correct spelling/grammar?

B1 LEVEL:
- Task type: email, letter, informal dialogue, short article.
- Focus: Can user organize ideas and explain reasons?
- Content: opinions, reasons, descriptions of experiences, advice.
- Grammar: Past/present/future tenses, complex sentences with subordinate clauses.
- Word count: 100-150 words.
- Example context: "Write a letter to a hotel manager about your recent stay. Explain what was good and suggest one improvement."
- Evaluation: Ideas clearly organized? Reasons provided? Appropriate tone? Good grammar?

B2 LEVEL:
- Task type: email, formal letter, article, report excerpt, dialogue.
- Focus: Can user argue, analyze, and write formally/informally?
- Content: complex opinions, analysis, hypothetical situations, formal requests.
- Grammar: Range of tenses, complex structures, nuanced language.
- Word count: 150-250 words.
- Example context: "Write a formal email to your employer proposing a change to the work schedule. Include background, benefits, and address potential concerns."
- Evaluation: Clear argument structure? Formal register maintained? Sophisticated vocabulary? Minimal errors?


Content Rules:
- Use realistic everyday scenarios: work, travel, shopping, family, hobbies, complaints, requests.
- Provide clear context and situation to make task meaningful.
- Avoid ambiguous or overly complex instructions.
- Ensure 3 content points are achievable within word limit.
- example_response should be well-written but concise.


Return ONLY the JSON array with 5 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Writing prompt generated successfully")
    return prompt




def get_speaking_prompt(topic, level="A1", item_id_start=1, interaction_type="interview"):
    """Speaking-specific generation prompt that returns an ARRAY of 6 speaking tasks."""
    logger.info(f"Generating speaking prompt: topic='{topic}', level={level}, item_id_start={item_id_start}, interaction_type={interaction_type}")
    
    logger.debug(f"Fetching vocabulary for topic '{topic}' at level {level}")
    vocab_list = fetch_vocab_from_vector_db(topic, level)
    logger.info(f"Retrieved {len(vocab_list)} vocabulary items for prompt generation")
    logger.debug(f"Vocabulary items: {vocab_list}")

    prompt = f"""
Task:
Generate EXACTLY 6 speaking tasks for CEFR {level}.
The output MUST be a SINGLE JSON ARRAY with 6 objects. No text before or after the JSON.


Inputs:
- vocab_list: {json.dumps(vocab_list, ensure_ascii=False)}
- topic: "{topic}"
- start_id: {item_id_start}
- interaction_type: "{interaction_type}" (can be: interview, dialogue, picture_description, roleplay, presentation, question_answer)


JSON ARRAY STRUCTURE (exact):
[
  {{
    "id": integer,
    "type": "{interaction_type}" | "dialogue" | "picture_description" | "roleplay" | "presentation" | "question_answer",
    "task_title": string,
    "task_description": string,
    "task_instructions": string,
    "prompt": string,
    "prompt_translation": string,
    "follow_up_questions": [string, string, string],
    "follow_up_translations": [string, string, string],
    "acceptable_response_length": "15-30 seconds" | "30-60 seconds" | "1-2 minutes",
    "vocabulary_required": [vocab from vocab_list],
    "grammar_structures_required": [string],
    "example_response": string,
    "example_response_translation": string,
    "evaluation_criteria": {{
        "fluency_pronunciation": string,
        "vocabulary_accuracy": string,
        "grammar_accuracy": string,
        "task_completion": string,
        "interaction_ability": string
    }},
    "metadata": {{
        "level": "{level}",
        "skill": "SPEAKING",
        "topic": "{topic}",
        "source": "generated",
        "timestamp": "{datetime.now().isoformat()}",
        "interaction_type": "{interaction_type}",
        "expected_duration_seconds": integer
    }}
  }},
  ...
]  <-- exactly 6 objects


CRITICAL REQUIREMENTS:
- Start IDs at {item_id_start} and increment sequentially.
- Task must incorporate 2-3 words from vocab_list in prompt.
- follow_up_questions: Generate 3 related follow-up questions to extend conversation.
- vocabulary_required: List specific vocab from vocab_list that student should use.
- grammar_structures_required: List expected grammar (e.g., "Present tense", "Modal verbs", "Past tense + present").
- example_response: Provide a 1-3 sentence model response at appropriate level.
- Responses can be transcribed audio later; focus on generating clear prompts and evaluation criteria.
- No explanations, no prose, no markdown — ONLY the JSON array.


Speaking Task Guidelines by Level:

A1 LEVEL:
- Types: interview, question_answer, basic dialogue, picture description (simple).
- Focus: Can user answer simple questions? Introduce themselves? Use basic phrases?
- Prompts: "Tell me your name and where you live", "What's your job?", "Do you like...?"
- Expected duration: 15-30 seconds.
- Grammar: Present tense mostly, simple sentences.
- Evaluation: Clear pronunciation? Understands question? Uses vocabulary learned?

A2 LEVEL:
- Types: dialogue, interview, question_answer, picture description (simple).
- Focus: Can user have short conversations? Describe simple situations? Handle familiar topics?
- Prompts: "Describe your typical day", "Tell me about your family", "What did you do last weekend?"
- Expected duration: 30-60 seconds.
- Grammar: Present + past tense, some complex sentences.
- Evaluation: Speaks with minimal pauses? Mostly clear pronunciation? Uses variety of tenses?

B1 LEVEL:
- Types: dialogue, roleplay, interview, picture_description, presentation (short).
- Focus: Can user express opinions? Explain reasons? Handle social interactions?
- Prompts: "Describe this picture and explain what is happening", "You're at a restaurant, order a meal", "What do you think about...? Why?"
- Expected duration: 1-2 minutes.
- Grammar: Multiple tenses, complex sentence structures, subordinate clauses.
- Evaluation: Fluent with occasional pauses? Natural pronunciation? Can express opinions?

B2 LEVEL:
- Types: dialogue (complex), roleplay, interview (advanced), presentation, discussion.
- Focus: Can user argue, hypothesize, handle unexpected turns? Speak fluidly?
- Prompts: "Discuss advantages and disadvantages of...", "What would you do if...?", "Explain your opinion on a controversial topic."
- Expected duration: 2-3 minutes per response.
- Grammar: Full range of tenses, sophisticated structures, subtle meaning.
- Evaluation: Very fluent with natural pauses? Native-like pronunciation? Nuanced arguments?


Content Rules:
- Use realistic everyday scenarios: introductions, shopping, travel, work, complaints, opinions, advice.
- Provide clear context for roleplay and dialogue tasks.
- Avoid abstract or overly philosophical topics for A1/A2.
- Make follow-up questions progressively harder (Q1 simple, Q3 challenging).
- example_response should demonstrate good pronunciation and appropriate level.


Evaluation_Criteria Guidelines:
- fluency_pronunciation: How smooth is speech? Clear pronunciation? Natural rhythm?
- vocabulary_accuracy: Does student use correct vocab? Appropriate to context?
- grammar_accuracy: Are verb tenses correct? Sentence structures sound natural?
- task_completion: Did student answer the question fully? Address all points?
- interaction_ability: Can student engage? Respond to follow-ups naturally?


Return ONLY the JSON array with 6 objects.
"""
    logger.debug(f"Generated prompt (length: {len(prompt)} characters)")
    logger.info("Speaking prompt generated successfully")
    return prompt




# Evaluation for writing and speaking
def evaluate_writing_response(writing_task, user_response):
    """Use LLM to evaluate if writing response is acceptable."""
    return f"""
You are a CEFR German exam grader at level {writing_task['metadata']['level']}.

TASK: {writing_task['task_description']}
INSTRUCTIONS: {writing_task['task_instructions']}
CONTENT POINTS REQUIRED: {', '.join(writing_task['content_points'])}
WORD COUNT RANGE: {writing_task['word_count_min']}-{writing_task['word_count_max']} words
EVALUATION CRITERIA: {json.dumps(writing_task['evaluation_criteria'], ensure_ascii=False)}

USER RESPONSE:
{user_response}

Evaluate the response on these criteria:
1. Task Completion: Are all 3 content points addressed?
2. Word Count: Is response within range?
3. Grammar: Are there major grammar errors?
4. Vocabulary: Is vocabulary appropriate and varied?
5. Organization: Is the text coherent and well-organized?

Respond in JSON:
{{
    "task_completed": true/false,
    "is_acceptable": true/false,
    "score_out_of_10": integer,
    "feedback": "constructive feedback",
    "errors": ["list of main errors"],
    "strengths": ["list of strengths"]
}}
"""


def evaluate_speaking_response(speaking_task, transcribed_audio_text):
    """Use LLM to evaluate if speaking response is acceptable."""
    return f"""
You are a CEFR German exam grader at level {speaking_task['metadata']['level']}.

TASK: {speaking_task['task_description']}
PROMPT: {speaking_task['prompt']}
EXPECTED DURATION: {speaking_task['acceptable_response_length']}
VOCABULARY REQUIRED: {', '.join(speaking_task['vocabulary_required'])}
GRAMMAR STRUCTURES: {', '.join(speaking_task['grammar_structures_required'])}
EVALUATION CRITERIA: {json.dumps(speaking_task['evaluation_criteria'], ensure_ascii=False)}

TRANSCRIBED AUDIO TEXT:
{transcribed_audio_text}

Evaluate the response on these criteria:
1. Task Completion: Did student answer the prompt? Address follow-up questions?
2. Fluency: Natural pace and rhythm? Minimal hesitations?
3. Pronunciation: Is pronunciation clear and mostly correct?
4. Grammar: Are verb tenses and sentence structures correct?
5. Vocabulary: Does student use required vocab? Appropriate level?

Respond in JSON:
{{
    "task_completed": true/false,
    "is_acceptable": true/false,
    "score_out_of_10": integer,
    "pronunciation_clarity": "clear" | "mostly_clear" | "needs_improvement",
    "fluency_level": "fluent" | "mostly_fluent" | "halting",
    "feedback": "constructive feedback",
    "errors": ["list of main errors"],
    "strengths": ["list of strengths"]
}}
"""