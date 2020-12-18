from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, BartForConditionalGeneration, RagConfig, DPRQuestionEncoder
import torch
from transformers.models.auto import AutoModel

config = RagConfig.from_pretrained ("facebook/rag-token-nq")
config.index_name = "legacy"
config.use_dummy_dataset = False
config.question_encoder.return_dict = True
print("==> load tokenizer")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
print("==> load retriever")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", config=config)
print("dataset info")
print(dir(retriever.index))
print("==> load generator")
# question encoder
# question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
# generator = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config)
# model = RagTokenForGeneration(config, question_encoder=question_encoder,generator=generator, retriever=retriever)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# input_dict = tokenizer.prepare_seq2seq_batch("USA president in 1999?", return_tensors="pt")
input_dict = tokenizer.prepare_seq2seq_batch("What kind of vehicle uses fire hydrant?", return_tensors="pt")
# input_dict = tokenizer.prepare_seq2seq_batch("what phylum does cat belong to?", return_tensors="pt")

print(input_dict.keys()) # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
input_ids = input_dict['input_ids']
print("==> encode")
question_hidden_states = model.question_encoder(input_ids)[0]
print(question_hidden_states.shape) # 1 x 768
print("==> retrieve")
docs_dict = model.retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
print(docs_dict["retrieved_doc_embeds"].shape) # 1 x 5 x 768
print(docs_dict.keys()) # 1 x 5 x 768
# print(docs_dict['doc_ids'])
all_docs = retriever.index.get_doc_dicts(docs_dict['doc_ids'].numpy())[0]
print(all_docs)
doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)# print(outputs.keys())  # odict_keys(['logits', 'doc_scores', 'past_key_values', 'context_input_ids', 'generator_enc_last_hidden_state'])
print("==> generate")
generated = model.generate(context_input_ids=docs_dict["context_input_ids"],
                           context_attention_mask=docs_dict["context_attention_mask"],
                           doc_scores=doc_scores, max_length=10)

generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
print(generated.shape) # 1 x 8  (B, T)
print(generated_string)