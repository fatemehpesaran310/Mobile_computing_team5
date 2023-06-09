from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelWithLMHead
from transformers import pipeline

with open('./story.txt', 'r') as f:
    story_lines = f.readlines()

story = ''.join(story_lines)
story = story.replace("\n", "").strip()

# Finding the main character
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
ner_results = nlp(story)
main_char = ner_results[0]['word']

# Save the main character to a file
with open('./main_character.txt', 'w') as f:
    f.write(main_char)