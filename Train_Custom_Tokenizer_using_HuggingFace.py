#!/usr/bin/env python
# coding: utf-8

# ## 1. Find a dataset
# 
# 

# In[1]:


get_ipython().getoutput('pip install datasets')


# In[2]:


import datasets


# In[3]:


all_ds = datasets.list_datasets()
print(len(all_ds))


# In[4]:


all_ds[:20]


# In[5]:


dataset = datasets.load_dataset('oscar', 'unshuffled_deduplicated_la')


# In[6]:


dataset


# In[7]:


dataset['train'][1]


# In[8]:


from tqdm.auto import tqdm  # for our loading bar

text_data = []
file_count = 0

for sample in tqdm(dataset['train']):
    # remove newline characters from each sample as we need to use exclusively as seperators
    sample = sample['text'].replace('\n', '')
    text_data.append(sample)
    if len(text_data) == 5_000:
        # once we hit the 5K mark, save to file
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1
# after saving in 5K chunks, we will have ~3808 leftover samples, we save those now too
with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(text_data))


# In[9]:


from pathlib import Path
paths = [str(x) for x in Path('/content/data/').glob('**/*.txt')]
paths


# ## 2. Train a tokenizer
# 
# We choose to train a byte-level Byte-pair encoding tokenizer (the same as GPT-2), with the same special tokens as RoBERTa. Let’s arbitrarily pick its size to be 52
# ,000.
# 
# We recommend training a byte-level BPE (rather than let’s say, a WordPiece tokenizer like BERT) because it will start building its vocabulary from an alphabet of single bytes, so all words will be decomposable into tokens (no more `<unk>` tokens!).
# 

# In[10]:


get_ipython().system('pip install transformers')


# In[11]:


from pathlib import Path
paths = [str(x) for x in Path(".").glob("**/*.txt")]
print(len(paths))


# In[12]:


from tokenizers import ByteLevelBPETokenizer
# initialize
tokenizer = ByteLevelBPETokenizer()
# and train
tokenizer.train(files=paths, vocab_size=30000, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])


# Now let's save files to disk

# In[13]:


#@title
get_ipython().system('mkdir LaRoBERTo')
tokenizer.save_model("LaRoBERTo")


# In[14]:


from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('LaRoBERTo')


# In[15]:


tokenizer('Sicut enim maius est illuminare quam lucere solum, ita maius est contemplata aliis tradere quam solum contemplari.', max_length=512, padding='max_length', truncation=True)


# ## 3. Train a language model from scratch
# 
# **Update:** This section follows along the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py) script, using our new [`Trainer`](https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py) directly. Feel free to pick the approach you like best.
# 
# > We’ll train a RoBERTa-like model, which is a BERT-like with a couple of changes (check the [documentation](https://huggingface.co/transformers/model_doc/roberta.html) for more details).
# 
# As the model is BERT-like, we’ll train it on a task of *Masked language modeling*, i.e. the predict how to fill arbitrary tokens that we randomly mask in the dataset. This is taken care of by the example script.
# 

# In[ ]:


# Check that we have a GPU
get_ipython().system('nvidia-smi')


# In[ ]:


# Check that PyTorch sees it
import torch
torch.cuda.is_available()


# ### We'll define the following config for the model

# In[ ]:


from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


# Now let's re-create our tokenizer in transformers

# In[ ]:


from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)


# Finally let's initialize our model.
# 
# **Important:**
# 
# As we are training from scratch, we only initialize from a config, not from an existing pretrained model or checkpoint.

# In[ ]:


from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)


# In[ ]:


model.num_parameters()
# => 84 million parameters


# ### Now let's build our training Dataset
# 
# We'll build our dataset by applying our tokenizer to our text file.
# 
# Here, as we only have one text file, we don't even need to customize our `Dataset`. We'll just use the `LineByLineDataset` out-of-the-box.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from transformers import LineByLineTextDataset\n\ndataset = LineByLineTextDataset(\n    tokenizer=tokenizer,\n    file_path="./oscar.eo.txt",\n    block_size=128,\n)\n')


# Like in the [`run_language_modeling.py`](https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_language_modeling.py) script, we need to define a data_collator.
# 
# This is just a small helper that will help us batch different samples of the dataset together into an object that PyTorch knows how to perform backprop on.

# In[ ]:


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# ### Finally, we are all set to initialize our Trainer

# In[ ]:


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


# ### Start training

# In[ ]:


get_ipython().run_cell_magic('time', '', 'trainer.train()\n')


# #### 🎉 Save final model (+ tokenizer + config) to disk

# In[ ]:


trainer.save_model("./EsperBERTo")


# ## 4. Check that the LM actually trained

# Aside from looking at the training and eval losses going down, the easiest way to check whether our language model is learning anything interesting is via the `FillMaskPipeline`.
# 
# Pipelines are simple wrappers around tokenizers and models, and the 'fill-mask' one will let you input a sequence containing a masked token (here, `<mask>`) and return a list of the most probable filled sequences, with their probabilities.
# 
# 

# In[ ]:


from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./EsperBERTo",
    tokenizer="./EsperBERTo"
)


# In[ ]:


# The sun <mask>.
# =>

fill_mask("La suno <mask>.")


# Ok, simple syntax/grammar works. Let’s try a slightly more interesting prompt:
# 
# 

# In[ ]:


fill_mask("Jen la komenco de bela <mask>.")

# This is the beginning of a beautiful <mask>.
# =>


# ## 5. Share your model 🎉

# Finally, when you have a nice model, please think about sharing it with the community:
# 
# - upload your model using the CLI: `transformers-cli upload`
# - write a README.md model card and add it to the repository under `model_cards/`. Your model card should ideally include:
#     - a model description,
#     - training params (dataset, preprocessing, hyperparameters), 
#     - evaluation results,
#     - intended uses & limitations
#     - whatever else is helpful! 🤓
# 
# ### **TADA!**
# 
# ➡️ Your model has a page on http://huggingface.co/models and everyone can load it using `AutoModel.from_pretrained("username/model_name")`.
# 
# [![tb](https://huggingface.co/blog/assets/01_how-to-train/model_page.png)](https://huggingface.co/julien-c/EsperBERTo-small)
# 

# If you want to take a look at models in different languages, check https://huggingface.co/models
# 
# [![all models](https://huggingface.co/front/thumbnails/models.png)](https://huggingface.co/models)
# 
