import tensorflow as tf
import tensorflow_datasets as tfds


class CommaTokenizer(tfds.features.text.Tokenizer):
    def tokenize(self, s):
        """ Splits a string into tokens. As we know our tokens are comma-separated, can just split on comma, which is
            a lot more efficient, as otherwise time complexity is linked to number of word in our vocabulary that
            contain non-alpha numeric characters.
        """
        s = tf.compat.as_text(s)
        toks = s.split(",")
        return toks


class CommaTokenTextEncoder(tfds.features.text.TokenTextEncoder):

    """ Only changes to super class are to enable support for CommaTokenizer """

    def __init__(self, vocab_list, oov_buckets=1, oov_token="UNK", lowercase=False, tokenizer=None, strip_vocab=True,
                 decode_token_separator=" "):
        super().__init__(vocab_list, oov_buckets, oov_token, lowercase, tokenizer, strip_vocab, decode_token_separator)
        # Do not need to pass reserved tokens as only used in Tokenizer, which we no longer use
        self._tokenizer = (tokenizer or CommaTokenizer(reserved_tokens=[]))

    @classmethod
    def load_from_file(cls, filename_prefix):
        filename = cls._filename(filename_prefix)
        vocab_lines, kwargs = cls._read_lines_from_file(filename)
        has_tokenizer = kwargs.pop("has_tokenizer", False)
        if has_tokenizer:
            kwargs["tokenizer"] = CommaTokenizer.load_from_file(filename)
        return cls(vocab_list=vocab_lines, **kwargs)
