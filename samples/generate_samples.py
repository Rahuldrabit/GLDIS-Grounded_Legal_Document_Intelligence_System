"""
Synthetic legal document generator for GLDIS sample data.
Run: python samples/generate_samples.py
"""
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

SAMPLE_1 = """COMMERCIAL SERVICE AGREEMENT

Case No. CV-2024-0847

This Commercial Service Agreement ("Agreement") is entered into as of January 15, 2024,
by and between PEARSON SPECTER LITT LLP, a limited liability partnership organized under
the laws of the State of New York, with its principal place of business at 579 Park Avenue,
New York, NY 10022 ("Client"); and AXIOM LEGAL SERVICES INC., a corporation organized
under the laws of Delaware, with its principal place of business at 1200 Corporate Drive,
Wilmington, DE 19801 ("Service Provider").

Section 1. SERVICES

1.1 Service Provider shall provide document review, legal research, and litigation support
services as described in Exhibit A.

1.2 Service Provider shall commence Services on February 1, 2024 and complete all
deliverables no later than December 31, 2024.

1.3 Service Provider must submit weekly progress reports every Friday by 5:00 PM EST.

Section 2. COMPENSATION

2.1 Client agrees to pay Service Provider a monthly retainer of USD 45,000.00, payable
within thirty (30) days of invoice receipt.

2.2 Client shall reimburse out-of-pocket expenses not to exceed USD 5,000.00 per month,
provided such expenses are pre-approved in writing.

Section 3. TERM AND TERMINATION

3.1 This Agreement shall commence on February 1, 2024 and continue through January 31, 2025.

3.2 Either party may terminate without cause upon sixty (60) days written notice.

3.3 Client may terminate for cause immediately if Service Provider materially breaches
any provision and fails to cure within fifteen (15) days of written notice.

Section 4. CONFIDENTIALITY

4.1 Service Provider shall not disclose any Confidential Information to any third party
without prior written consent. This obligation survives termination for five (5) years.

Section 5. GOVERNING LAW

5.1 This Agreement shall be governed by the laws of the State of New York.

5.2 Any disputes shall be resolved by binding arbitration in New York County, New York.

Signed:
Harvey Specter, Managing Partner — PEARSON SPECTER LITT LLP — January 15, 2024
Dana Scott, CEO — AXIOM LEGAL SERVICES INC. — January 15, 2024
"""

SAMPLE_2 = """N0TICE 0F CLAIM AND DEM AND F0R ARBITRATI0N
[SCANNED D0CUMENT - PARTIAL ILLEGIBILITY]

Case N0. ARB-2O24-I123
Date: March 04, 2024

T0: G10ba1 F1nanc1a1 Partners LLC, 440 Madis0n Avenue, New Y0rk, NY 10022
FR0M: Hartw00d Capital Management Inc., C/0 Rach1e Zane Esq., 888 F1fth Avenue, NY 10065

RE: D1spute - Investment Management Agreement dated June 12, 2022

FACTS:
1. On 0r ab0ut July 15, 2023, Resp0ndent failed t0 execute a scheduled
   p0rtf0lio rebalancing required under Sect10n 4.2 0f the IMA.
2. Claimant suffered financial l0sses estimated at USD 2,350,000.00.
3. Resp0ndent was n0tified 0f the breach 0n August 2, 2023 but failed
   t0 cure within the required 30-day cure per10d.
4. Claimant demands c0mpensati0n 0f USD 2,350,000.00 plus interest at 8% per annum.

RELIEF S0UGHT:
a) C0mpensati0n 0f USD 2,350,000.00
b) Interest at 8% per annum from July 15, 2023
c) Reimb0rsement 0f legal fees and arbitrati0n c0sts

DEADL1NE F0R RESP0NSE: April 4, 2024

Rach1e Zane, Esq. | Bar N0. NY-294751 | March 04, 2024
"""

SAMPLE_3 = """INTERNAL CASE NOTES — CONFIDENTIAL
[PARTIALLY HANDWRITTEN - TRANSCRIPTION UNCERTAIN]

Matter: Thompson v. Langford Industries
Case No.: CV-2023-4492
Assigned Attorney: Mike Ross
Date: September 18, 2023

KEY FACTS:
- Plaintiff: James Thompson, former VP of Operations at Langford Industries
- Defendant: Langford Industries Corp., Delaware corporation
- Claim: Wrongful termination + breach of employment contract

TIMELINE:
  * January 3, 2022 - Thompson hired under 3-year contract at USD 380,000/year + 20% bonus
  * [UNCERTAIN] August [?] 2023 - Thompson terminated
  * Termination reason: "performance issues" — Thompson disputes, claims retaliation for fraud report

OBLIGATIONS:
- Langford SHALL provide 90 days severance per Section 7.1
- Langford MUST provide COBRA benefits for 18 months
- Thompson required to return all company property within 14 days

FINANCIAL EXPOSURE:
  * Remaining contract value: approx USD 570,000
  * Severance owed: USD 95,000
  * Potential punitive damages: [UNCERTAIN] USD 500,000 to USD 1,200,000
  * Legal fees to date: USD 28,450

JURISDICTIONS: State of Delaware; State of New York; US District Court SDNY

DEADLINES:
  - File motion for discovery by October 15, 2023
  - Depose HR Director before November 30, 2023
  - Settlement conference: December 5, 2023

Evidence: Employment contract, HR termination memo, internal fraud report,
email chain CEO/HR (July-Aug 2023), performance review records.

Status: ACTIVE - Negotiation phase
"""

SAMPLE_4 = """COMMERCIAL LEASE AGREEMENT

Case Reference: RL-2024-0391

LANDLORD: Midtown Properties LLC, 1500 Broadway Suite 900, New York NY 10036
          Contact: Louis Litt, Property Manager

TENANT:   Sterling Cooper Legal Group LLP, 620 Fifth Avenue, New York NY 10020

PREMISES: Suite 1200-1400, 620 Fifth Avenue, New York NY 10020 — 8,500 sq ft

LEASE TERM: March 1, 2024 through February 28, 2029 (Five years)

FINANCIAL SCHEDULE:
  Year 2024: USD 42,500/month | USD 510,000/year
  Year 2025: USD 44,625/month | USD 535,500/year
  Year 2026: USD 46,856/month | USD 562,275/year
  Year 2027: USD 49,199/month | USD 590,389/year
  Year 2028: USD 51,659/month | USD 619,908/year

SECURITY DEPOSIT: USD 127,500 (three months) due on signing
FIRST MONTH RENT: USD 42,500 due March 1, 2024

Article 3 - Use of Premises
Tenant shall use the Premises solely for law firm operations. No illegal use permitted.

Article 5 - Alterations
Tenant must obtain prior written consent before alterations exceeding USD 10,000.

Article 7 - Insurance
Tenant is required to maintain:
  a) Commercial General Liability: minimum USD 2,000,000 per occurrence
  b) Property Insurance: replacement cost coverage
  c) Workers Compensation: as required by New York law
Certificates due no later than February 15, 2024.

Article 9 - Default
Rent unpaid after five (5) days incurs 5% late fee. Failure to pay within
thirty (30) days entitles Landlord to terminate upon written notice.

Article 11 - Renewal
Tenant has one option to renew for five (5) additional years, provided 180-day
prior written notice and no existing default. Market rate rent applies.

Governing Law: State of New York | Venue: New York County Supreme Court

Louis Litt — Midtown Properties LLC — February 10, 2024
Name Redacted, Partner — Sterling Cooper Legal Group LLP — February 12, 2024
"""


def generate_all():
    samples = [
        ("sample_01_commercial_agreement.txt", SAMPLE_1, "Clean commercial contract"),
        ("sample_02_arbitration_notice.txt",   SAMPLE_2, "Noisy/OCR-artifact document"),
        ("sample_03_case_notes.txt",           SAMPLE_3, "Handwritten-style case notes"),
        ("sample_04_lease_agreement.txt",       SAMPLE_4, "Multi-page lease with tables"),
    ]
    OUTPUT_DIR.mkdir(exist_ok=True)
    for filename, content, desc in samples:
        (OUTPUT_DIR / filename).write_text(content.strip(), encoding="utf-8")
        print(f"[OK] {filename}  ({desc})")
    print(f"\n{len(samples)} sample documents written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()
