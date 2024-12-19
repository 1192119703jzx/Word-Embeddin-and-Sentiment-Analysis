# Word-Embedding-and-Sentiment-Analysis
Zixin Jiang

## How to run
- We have three kinds of embedding: static embedding, advaned static embedding, and contextual embedding
- corpus.py, model.py, run.py is for static embedding
- corpus_advanced.py is for advaned static embedding and corpus_context.py is for contextual embedding
- model_enhenced.py and run_context.py is for both advaned static embedding and contextual embedding

1. run corpus.py first to generate pre-process train, dev, test data class. The output will be save as files for later usage and avoid redundant pre-process procedual each time.
2. run rn.py to do the experiment. Change the parameters as you wish. Based on the model/embedding you want to use, comment or uncommnet certain lines of code according to the instructions inside the files.
3. The output will show the training loss, validation loss, and validation accuracy for each epoch.
