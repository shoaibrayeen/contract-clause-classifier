from classification.segment import segment_contract

CONTRACT = """MASTER SERVICES AGREEMENT

This Master Services Agreement is entered into as of January 1, 2024 by and between Acme Corp and Beta LLC, collectively the parties, who agree as follows below in this document.

ARTICLE I
DEFINITIONS

1. Definitions. Capitalized terms used in this Agreement have the meanings given in this Section, and such meanings apply equally to singular and plural forms of the terms defined herein.

2. Confidentiality. Each party shall keep confidential all Confidential Information of the other party and shall not disclose it to any third party without prior written consent, except as required by law.
(a) The obligations in this Section survive termination for five years following the end of the term of this Agreement.
(b) Return of materials shall occur promptly upon written request by the disclosing party at any time.

3. Termination. Either party may terminate this Agreement upon thirty days written notice to the other party if the other party materially breaches this Agreement and fails to cure.

4. Governing Law. This Agreement shall be governed by and construed in accordance with the laws of the State of New York, without regard to its conflict of laws principles.
"""


def test_splits_numbered_sections():
    provs = segment_contract(CONTRACT)
    texts = [p.text for p in provs]
    assert any(t.startswith("2. Confidentiality") or "2. Confidentiality" in t for t in texts)
    assert any("Governing Law" in t for t in texts)
    # Sub-items stay with their parent section.
    conf = next(t for t in texts if "2. Confidentiality" in t)
    assert "(a) The obligations" in conf and "(b) Return" in conf
    # Indices are sequential.
    assert [p.index for p in provs] == list(range(len(provs)))


def test_heading_glued_to_next_paragraph():
    provs = segment_contract(CONTRACT)
    assert not any(p.text.strip() in {"ARTICLE I", "DEFINITIONS", "ARTICLE I DEFINITIONS"} for p in provs)


def test_long_text_is_split():
    long = " ".join(["The Supplier shall deliver the goods on time."] * 200)
    provs = segment_contract(long, max_words=100)
    assert len(provs) > 1
    assert all(len(p.text.split()) <= 100 for p in provs)


def test_empty_input():
    assert segment_contract("   \n  ") == []


def test_hard_wrapped_pdf_lines_are_joined():
    text = "1. Notices. All notices under this Agreement shall be in writing and\nshall be deemed given when delivered personally or by overnight courier to the\naddress set forth above for each party hereto.\n\n2. Assignment. Neither party may assign this Agreement without the prior written consent of the other party, which shall not be unreasonably withheld."
    provs = segment_contract(text)
    assert len(provs) == 2
    assert "\n" not in provs[0].text
