from  transformers  import  AutoTokenizer, AutoModelWithLMHead, pipeline

model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

with open('./main_character.txt','r') as f:
   main_char = f.read()

with open('./story.txt', 'r') as f:
    story_lines = f.readlines()

story = ''.join(story_lines)
story = story.replace("\n", "").strip()


questions = [
    f"Can you describe {main_char}'s appearance?",
    f"Tell me about {main_char}'s physical features.",
    f"What are the distinguishing characteristics of {main_char}?",
]

descriptions = []
for question in questions:
    input_text = f"question: {question} context: {story}"
    encoded_input = tokenizer(
        [input_text],
        return_tensors='pt',
        max_length=512,
        truncation=True
    )
    output = model.generate(
        input_ids=encoded_input.input_ids,
        attention_mask=encoded_input.attention_mask
    )
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    descriptions.append(description)

main_char_des = " ".join(descriptions)
main_char_des = main_char + ' is ' + main_char_des

print(main_char_des)

with open('description_of_main_character.txt', 'w') as f:
    f.write(main_char_des)




