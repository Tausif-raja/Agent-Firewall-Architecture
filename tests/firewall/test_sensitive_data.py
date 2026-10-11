import pytest

from firewall.sensitive_data_engine import SensitiveDataEngine

ENGINE = SensitiveDataEngine(use_presidio=False)


def types(text):
    return {f.entity_type for f in ENGINE.detect(text)}


@pytest.mark.parametrize("text,expected", [
    ("mail me at jane.doe@example.org", "EMAIL_ADDRESS"),
    ("card 4111 1111 1111 1111 exp 12/27", "CREDIT_CARD"),
    ("IBAN DE89 3704 0044 0532 0130 00", "IBAN"),
    ("ssn 123-45-6789", "US_SSN"),
    ("call +1 415 555 0134", "PHONE_NUMBER"),
    ("server at 192.168.10.4", "IP_ADDRESS"),
    ("key sk-or-v1-" + "a" * 64, "OPENROUTER_API_KEY"),
    ("token gsk_" + "b" * 40, "GROQ_API_KEY"),
    ("my password is hunter2!", "PASSWORD"),
    ("postgresql://admin:s3cret@db.internal:5432/app", "CONNECTION_STRING"),
    ("I was diagnosed with type 2 diabetes.", "HEALTH_DATA"),
])
def test_detects(text, expected):
    assert expected in types(text)


@pytest.mark.parametrize("text", [
    "What is the capital of France?",
    "Order 12345 costs 99.50 and ships in 3 days",
    "Upgrade from version 1.2.3 to 2.0.1",
    "The meeting is on 2024-05-01 at 10:30",
    "card number 4111 1111 1111 1112",       # fails Luhn
    "IP 999.300.1.1 is not valid",
    "Reset my password please",
])
def test_no_false_positives(text):
    assert types(text) == set(), types(text)


def test_tokenize_is_reversible_per_session_and_stable():
    e = SensitiveDataEngine(use_presidio=False)
    out, findings, block = e.sanitize_or_pseudonymize("a@x.io and a@x.io and b@y.io", "s1", "tokenize")
    assert out == "<EMAIL_ADDRESS_1> and <EMAIL_ADDRESS_1> and <EMAIL_ADDRESS_2>"
    assert not block and len(findings) == 3
    assert e.unmask_egress("Reply to <EMAIL_ADDRESS_2>", "s1") == "Reply to b@y.io"
    assert e.unmask_egress("Reply to <EMAIL_ADDRESS_2>", "other-session") == "Reply to <EMAIL_ADDRESS_2>"


def test_secrets_never_enter_vault():
    e = SensitiveDataEngine(use_presidio=False)
    out, _, _ = e.sanitize_or_pseudonymize("use gsk_" + "c" * 40, "s", "tokenize")
    assert out == "use <GROQ_API_KEY>"
    assert e.unmask_egress(out, "s") == out


def test_block_mode():
    _, findings, block = ENGINE.sanitize_or_pseudonymize("ssn 123-45-6789", "s", "block")
    assert block and findings


def test_finding_dict_never_contains_raw_value():
    f = ENGINE.detect("mail jane.doe@example.org")[0].to_dict()
    assert "jane.doe@example.org" not in str(f)
    assert f["snippet"] == "ja***rg"


def test_ropa_record():
    rec = ENGINE.generate_ropa_record(ENGINE.detect("I was diagnosed with asthma. mail a@b.co"), "s")
    assert rec["special_categories_art9"] == ["HEALTH_DATA"]
    assert None not in rec["technical_measures"]
