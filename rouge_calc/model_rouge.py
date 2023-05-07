import numpy as np
from datasets import load_dataset
from summarizer import Summarizer
from rouge import Rouge
from tqdm import tqdm
import pickle
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torch

#Rouge: https://github.com/pltrdy/rouge
def calc_rouge(summ_model, test_input, test_label, ratio_val, rouge):
    scores_dict = {'rouge-1': {'r': np.zeros((len(test_input), )), 'p': np.zeros((len(test_input), )), 'f': np.zeros((len(test_input), ))}, 
               'rouge-2': {'r': np.zeros((len(test_input), )), 'p': np.zeros((len(test_input), )), 'f': np.zeros((len(test_input), ))}, 
               'rouge-l': {'r': np.zeros((len(test_input), )), 'p': np.zeros((len(test_input), )), 'f': np.zeros((len(test_input), ))}}
    
    for i in tqdm(range(len(test_input))):
        hypothesis = model(test_input[i], ratio = ratio_val)
        reference = test_label[i]
        scores = rouge.get_scores(hypothesis, reference)

        for j in scores_dict.keys():
            for k in scores_dict[j].keys():
                scores_dict[j][k][i] = scores[0][j][k]
    
    return scores_dict

def calc_rouge_pred(test_pred, test_label, rouge):
    scores_dict = {'rouge-1': {'r': np.zeros((len(test_pred), )), 'p': np.zeros((len(test_pred), )), 'f': np.zeros((len(test_pred), ))}, 
               'rouge-2': {'r': np.zeros((len(test_pred), )), 'p': np.zeros((len(test_pred), )), 'f': np.zeros((len(test_pred), ))}, 
               'rouge-l': {'r': np.zeros((len(test_pred), )), 'p': np.zeros((len(test_pred), )), 'f': np.zeros((len(test_pred), ))}}
    
    for i in tqdm(range(len(test_pred))):
        hypothesis = test_pred[i]
        reference = test_label[i]
        scores = rouge.get_scores(hypothesis, reference)

        for j in scores_dict.keys():
            for k in scores_dict[j].keys():
                scores_dict[j][k][i] = scores[0][j][k]
    
    return scores_dict

def print_txt(dict_scores, txt_name):
    #https://www.pythontutorial.net/python-basics/python-write-text-file/
    #https://www.geeksforgeeks.org/reading-writing-text-files-python/
    with open(txt_name, 'w') as f:
        for i in dict_scores.keys():
            for j in dict_scores[i].keys():
                f.write(i + " " + j + ": " + str(np.mean(dict_scores[i][j]))+"\n")
    f.close()

#https://colab.research.google.com/github/BritneyMuller/colab-notebooks/blob/master/Easy_Text_Summarization_with_BART.ipynb#scrollTo=-DcmOk-0UPvv
#Code taken and/or modified from colab file released by google
def text_summarize(tokenizer, model, text, device, num_beams, length_penalty, max_length, min_length, no_repeat_ngram_size):

  text = text.replace('\n','')
  text_input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=1024)['input_ids'].to(device)
  summary_ids = model.generate(text_input_ids, num_beams=int(num_beams), length_penalty=float(length_penalty), max_length=int(max_length), min_length=int(min_length), no_repeat_ngram_size=int(no_repeat_ngram_size))           
  summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
  return summary_txt


def text_summarize_bart_param(tokenizer, model, text, device, num_beams, length_penalty, max_length, min_length, no_repeat_ngram_size, early_stopping):

  text = text.replace('\n','')
  text_input_ids = tokenizer.batch_encode_plus([text], return_tensors='pt', max_length=512, truncation=True, padding=True)['input_ids'].to(device)
  summary_ids = model.generate(text_input_ids, num_beams=int(num_beams), length_penalty=float(length_penalty), max_length=int(max_length), min_length=int(min_length), no_repeat_ngram_size=int(no_repeat_ngram_size), early_stopping=early_stopping)           
  summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return summary_txt


if __name__=='__main__':
    dataset = load_dataset("kmfoda/booksum")
    test_input = dataset["test"]["chapter"]
    test_label = dataset["test"]["summary_text"]
    rouge = Rouge()
    device = 'cuda'

    #------Length of each test_input----------------------------------------
    test_input_length = np.zeros((len(test_input), ))
    for i in range(len(test_input)):
        test_input_length = len(test_input[i])
    #-----------------------------------------------------------------------




    # #------bert_simple-------------------------------------------------------
    # model = Summarizer()
    
    # ######-------ratio = 0.2-------------------------------------------------
    # bert_rouge_02 = calc_rouge(model, test_input, test_label, 0.2, rouge)
    # print_txt(bert_rouge_02, 'bert_simple_rouge_02.txt')
    # pickle.dump(bert_rouge_02, open('dict_pickle/bert_rouge_02.dict', 'wb'))
    # #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object

    # #####-------ratio = 0.5-------------------------------------------------
    # bert_rouge_05 = calc_rouge(model, test_input, test_label, 0.5, rouge)
    # print_txt(bert_rouge_05, 'bert_simple_rouge_05.txt')
    # pickle.dump(bert_rouge_05, open('dict_pickle/bert_rouge_05.dict', 'wb'))
    
    # #####-------ratio = 0.7-------------------------------------------------
    # bert_rouge_07 = calc_rouge(model, test_input, test_label, 0.7, rouge)
    # print_txt(bert_rouge_07, 'bert_simple_rouge_07.txt')
    # pickle.dump(bert_rouge_07, open('dict_pickle/bert_rouge_07.dict', 'wb'))





    #---------bart-large-cnn------------------------------------------------
    #https://github.com/huggingface/transformers
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    model.to(device)

    #####------Bart_pred_1--------------------------------------------------
    # Bart_pred_1 = []

    # for i in tqdm(range(len(test_input))):
    #     pred = text_summarize(tokenizer, model, test_input[i], device, 4, 2.0, 142, 56, 3)
    #     Bart_pred_1.append(pred)

    # Bart_rouge_1 = calc_rouge_pred(Bart_pred_1, test_label, rouge)
    # print_txt(Bart_rouge_1, 'Bart_rouge_1.txt')
    # pickle.dump(Bart_rouge_1, open('dict_pickle/Bart_rouge_1.dict', 'wb'))
    #####------Bart_pred_2--------------------------------------------------
    Bart_pred_2 = []

    for i in tqdm(range(len(test_input))):
        pred = text_summarize_bart_param(tokenizer, model, test_input[i], device, 4, 2.0, 142, 56, 3, True)
        Bart_pred_2.append(pred)

    Bart_rouge_2 = calc_rouge_pred(Bart_pred_2, test_label, rouge)
    print_txt(Bart_rouge_2, 'Bart_rouge_2.txt')
    pickle.dump(Bart_rouge_2, open('dict_pickle/Bart_rouge_2.dict', 'wb'))
    




    