from pipelines.bm25.tokenizer import tokenize


def test_preserves_numbers_and_decimals():
    assert tokenize("Revenue grew 10.5 percent in fiscal 2025.") == \
        ["revenue", "grew", "10.5", "percent", "in", "fiscal", "2025"]


def test_preserves_percent_sign():
    assert tokenize("Gross margin was 44.5%.") == ["gross", "margin", "was", "44.5%"]


def test_preserves_currency_symbol():
    assert tokenize("Total net sales were $394.3 billion.") == \
        ["total", "net", "sales", "were", "$394.3", "billion"]


def test_strips_table_pipes_but_keeps_cell_values():
    assert tokenize("| Revenue | $500 million |") == ["revenue", "$500", "million"]


def test_trailing_comma_not_absorbed_into_number():
    assert tokenize("In 2025, revenue grew") == ["in", "2025", "revenue", "grew"]
    assert tokenize("Revenue of $1,234, up 5%.") == ["revenue", "of", "$1,234", "up", "5%"]


def test_no_stemming_applied():
    assert tokenize("sales sales") == ["sales", "sales"]


def test_empty_string_returns_empty_list():
    assert tokenize("") == []


def test_whitespace_only_returns_empty_list():
    assert tokenize("   \n\t  ") == []
