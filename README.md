# TV-Script-Generation
Recurrent networks to generate new text from TV scripts; LSTM(long short-term memory) networks with PyTorch.
Applied RNN to generate a new, "fake" Seinfeld TV scripts using RNNs from the reconized patterns while training the Seinfeld dataset of scripts from 9 seasons. 
Dataset Stats from Kaggle [Seinfeld Chronicles](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv)
## Background 
* RNN , Recurrent Neural Networks (RNNs), which are machine learning models that are able to recognize and act on sequences of inputs.
* LSTM: Long Short-Term Memory Networks (LSTM), which forms a memory about a sequence of inputs, over time.

## Implementation Approach
### Dataset
Dataset Stats from Kaggle [Seinfeld Chronicles](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv)
* the number of unique words: 46k
* Number of lines: 109k
* Average number of words in each line: 5.54

### Pre-processing Data 
The function create_lookup_tables create two dictionaries:
- vocab_to_int : a dictionary to go from the words to an id
- int_to_vocab : a dictionary to go from the id to word
The function create_lookup_tables return these dictionaries as a tuple (vocab_to_int, int_to_vocab)

### Batching Data
1. Data into sequences
The function batch_data breaks up word id's into the appropriate sequence lengths, such that only complete sequence lengths are constructed.
2. Cpmverting Data into TensorDataset
In the function batch_data, data is converted into Tensors and formatted with TensorDataset.
```python
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
```
3. Batching Data
Finally, batch_data returns a DataLoader for the batched training data.

### RNN & LSTM Implementation 
![RNN/LSTM](decoder.png)
- The RNN class has complete __init__, forward , and init_hidden functions.
 * __init__ - The initialize function.
 * init_hidden - The initialization function for an LSTM/GRU hidden state
 * forward - Forward propagation function.
- This RNN model implemented LSTM as memory cell and fully-connected layer to generate new vocabs
- Lastly, trained recurrent neural networks to generate new characters,words, and bodies of text.
The initialize function creates the layers of the neural network and save them to the class. 
The forward propagation function uses these layers to run forward propagation and generates an output and a hidden state.
The output of this model is the last batch of word scores after a complete sequence has been processed. For each input sequence of words, the output is the word scores for a single, next word.
```python
lstm_output = lstm_output.contiguous().view(-1, self.hidden_dim)
# the last batch of word scores by shaping the output of the final, fully-connected layer
# reshape into (batch_size, seq_length, output_size)
output = output.view(batch_size, -1, self.output_size)
# get last batch
out = output[:, -1]
```

### forward and backpropgation
forward and backpropagation is implemented.  This returns the average loss over a batch and the hidden state returned by a call to RNN.
```python
loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
```

### RNN Training - Hyperparameter Tunning 
- Model Parameter 
    * embedding_dim: 50~200 not over 500 significantly smaller than the size of the vocabulary(46k)
    * hidden_dim : Hidden dimension, number of units in the hidden layers of the RNN
    * n_layers :the number of layers/cells in a RNN/LSTM between 1-3 --> selected 2 
    * sequence_length: the size of the length of sentences to look at before generating the next word
    * vocab_size : the number of uniqe tokens in training data vocabulary, 46k
    * output_size : desired size of the output
 
- Training Parameter 
    * learning_rate: learning rate for Adam optimizer , started 0.001
    * num_epochs :the number of iteration to train in order to get near a minimum in the training loss
    * batch_size: large enough to train efficiently, but small enough to fit the data in memory but not to over GPU capacity. tried 64, 128
 
### Embeddings & Tockenization/Punctuation processing 
 embeddings in neural networks by implementing a word2vec model that converts words into a representative vector of numerical values.


