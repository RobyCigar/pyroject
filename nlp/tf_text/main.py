import tensorflow as tf
import tensorflow_text as tf_text


def preprocess(vocab_lookup_table, example_text):

  # Normalize text
  tf_text.normalize_utf8(example_text)

  # Tokenize into words
  word_tokenizer = tf_text.WhitespaceTokenizer()
  tokens = word_tokenizer.tokenize(example_text)

  # Tokenize into subwords
  subword_tokenizer = tf_text.WordpieceTokenizer(
      vocab_lookup_table, token_out_type=tf.int64)
  subtokens = subword_tokenizer.tokenize(tokens).merge_dims(1, -1)

  # Apply padding
  padded_inputs = tf_text.pad_model_inputs(subtokens, max_seq_length=16)
  return padded_inputs
