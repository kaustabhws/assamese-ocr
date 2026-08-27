from axomiya_ocr.data.vocab import Vocabulary


def test_vocab_has_blank_and_required_assamese_characters(tmp_path) -> None:
    vocab = Vocabulary.build(["অসমীয়া", "ভাষা"])
    assert vocab.size == len(vocab.characters) + 1
    assert "ৰ" in vocab.characters
    assert "ৱ" in vocab.characters
    assert 0 not in vocab.char_to_id.values()
    path = tmp_path / "vocab.json"
    vocab.save(path)
    assert Vocabulary.load(path) == vocab


def test_ctc_decoding_collapses_repeat_and_blank() -> None:
    vocab = Vocabulary(("অ", "স"))
    assert vocab.decode_ctc([0, 1, 1, 0, 2, 2]) == "অস"

