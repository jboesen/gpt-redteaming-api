import torch
from transformers import (
  GPT2LMHeadModel,
  GPT2Tokenizer,
)
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

# https://towardsdatascience.com/text-generation-with-python-and-gpt-2-1fecbff1635b
def generate_text_greedy(seq, num_results = 1, output_attentions = False):
    """Generates text from sequence using greedy search."""
    # we pass a maximum output length of 200 tokens
    outputs_dict = model.generate(
        seq,
        max_length=150,
        do_sample=True,
        output_attentions=output_attentions,
        num_return_sequences=num_results,
        return_dict_in_generate=True)
    text = [tokenizer.decode(prediction,
        skip_special_tokens=True,
        output_attentions=True
        ) for prediction in outputs_dict['sequences']]
    return (text, outputs_dict['attentions']) if output_attentions else text

def generate_text_beam(encoded_seq, num_results = 1, output_attentions = False):
    """Generates text from sequence using beam search."""
  # keys: sequences, attentions
    outputs_dict = model.generate(
        encoded_seq,
        num_beams=num_results,
        max_length=500,
        num_return_sequences=num_results,
        no_repeat_ngram_size=1,
        remove_invalid_values=True,
        output_attentions=output_attentions,
        return_dict_in_generate=True,
    )
    text = [tokenizer\
            .decode(prediction,
                    skip_special_tokens=True,
                    output_attentions=True)
            for prediction in outputs_dict['sequences']]
    return (text, outputs_dict['attentions']) if output_attentions else text

def format_attn(attention):
    """Collapses transformer attention output 6-D output into 3-D output,
    returns numpy arr shape num_heads × sequence_length × sequence_length"""
  # attention is (? × layers × tensor(batch_size, num_heads, sequence_length, sequence_length)
    formatted_heads = torch.zeros(tuple(attention[0][0].shape))
    for layer in attention[0]:
        # will always have batch size of 1, so take first elem
        formatted_heads += layer[0]
    # average out
    heads = formatted_heads.squeeze(0).numpy()
    averaged = np.add.reduce(np.copy(heads)) / 12
    combined = np.concatenate((heads, np.expand_dims(averaged, axis=0)))
    return combined

def get_embedding(word):
    """Returns model's word embedding vector"""
    text_index = tokenizer.encode(word, add_prefix_space=True)
    return model.transformer.wte.weight[text_index,:]

def get_aggregated_completions(prompt, num_seqs):
    """Formats and calls prediction model with beam and greedy search,
        returning attention and prediction"""
    num_seqs = int(num_seqs)
    encoded_seq = tokenizer.encode(prompt, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encoded_seq[0])
    greedy_generated, greedy_attn = generate_text_greedy(\
        encoded_seq, num_results=num_seqs//2, output_attentions=True)
    beam_generated = generate_text_beam(encoded_seq, num_results=num_seqs-num_seqs//2)
    return {'greedy': greedy_generated,
            'beam': beam_generated,
            'attention': format_attn(greedy_attn),
            'tokens': tokens}

# get_aggregated_completions('my name is jeff, I dont know what I should write', 2)

# def __main__():
#     encoded_seq = tokenizer.encode('what is up my name is jeff', return_tensors='pt')
#     beam_generated, beam_attn = generate_text_beam(encoded_seq, num_results=1)
#     format_attn(beam_attn)

# __main__()
