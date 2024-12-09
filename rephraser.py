from sentence_transformers import SentenceTransformer, util
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

# paraphrase-multilingual-MiniLM-L12-v2
def __cosine_similarity(sentences):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    return cosine_matrix

# t5_paraphraser
def __t5_paraphraser(sentence, amount=10, seed=time.time(), max_length=0, top_k=50, top_p=0.95):
    amount = amount if amount > 0 else 10
    seed = seed if seed > 0 else time.time()
    max_length = max_length if max_length > 0 else sentence.__len__()*2
    top_k = top_k if top_k > 0 else 50
    top_p = top_p if top_p > 0 else 0.95
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = T5ForConditionalGeneration.from_pretrained('t5_paraphraser')
    tokenizer = T5Tokenizer.from_pretrained('t5_paraphraser')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("device ",device)
    model = model.to(device)
    text =  "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    beam_outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        do_sample=True,
        max_length= max_length,
        top_k= top_k,
        top_p= top_p,
        early_stopping=True,
        num_return_sequences=amount
    )

    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs

def rephrase(sentence, amount=10, seed=time.time(), max_length=0, top_k=50, top_p=0.95):
    sentences = __t5_paraphraser(sentence, amount, seed, max_length, top_k, top_p)
    cosine = [sentence] + sentences
    cosine_matrix = __cosine_similarity(cosine)
    sentencesWithCosine = []
    for i in range(1, len(cosine)):
        sentencesWithCosine.append((cosine[i], cosine_matrix[0][i].item()))
    sentencesWithCosine.sort(key=lambda x: x[1], reverse=True)
    return sentencesWithCosine
