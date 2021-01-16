# TV-Script-Generation
This project applied RNN to generate a new, "fake" Seinfeld TV scripts using RNNs from the recognized patterns while training the Seinfeld dataset of scripts from 9 seasons TV series. 
Dataset Stats are available from Kaggle [Seinfeld Chronicles](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv)
## Background 
* RNN(Recurrent Neural Networks): a deep learning models that are able to recognize and act on sequences of inputs.
* LSTM: Long Short-Term Memory Networks (LSTM), which forms a memory about a sequence of inputs, over time.

## Implementation Approach
### Dataset
Dataset Stats from Kaggle [Seinfeld Chronicles](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv)
* the number of unique words: 46k
* Number of lines: 109k
* Average number of words in each line: roughly 5.54 unique vocabularies per line 

### Pre-processing Data 
1. The function create_lookup_tables create two dictionaries:
- vocab_to_int : a dictionary to go from the words to an id
- int_to_vocab : a dictionary to go from the id to word
The function create_lookup_tables return these dictionaries as a tuple (vocab_to_int, int_to_vocab)
2. Tokenize Punctuation 
After splitting the script into a word array using spaces as delimiters punctuations like periods and exclamation marks should be processed in order to create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.
Implemented the function token_lookup to return a dictionary that is used to tokenize symbols like "!" into "||Exclamation_Mark||" in a key and a value pair.
3. DataLoader 
Used TensorData with DataLoader in Pytorch to handle batching , suffling and iterations. 
```python
data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
```

### Batching Data
1. Data into sequences
The function batch_data breaks up word id's into the appropriate sequence lengths, such that only complete sequence lengths are constructed.
2. Converting Data into TensorDataset
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
- Lastly, trained recurrent neural networks to generate new words, and bodies of text.
The initialize function creates the layers of the neural network and save them to the class. 
The forward propagation function uses these initialized layers to run forward propagation and generates an output and a hidden state.
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
All parameters need to be optimized for better performance and efficient computing to fit in the memory, however, I'd like to mention one parameter , sequence_length, since it determine the size of the long range dependencies that a model can learn. The more, the better new sentences look make sense. In practice, to generate sentences from novel, 100 sequence were used/recommended but because of the memory warning , I set it to 10 since average word per line was 5.5/line

### Model performance 
The loss decreased during training and reached a value lower than 3.5. 

### Improvement 
* Use validation data to choose the best model
* Initialize model weights, especially the weights of the embedded layer to encourage model convergence
* Use topk sampling to generate new words



