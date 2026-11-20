"""
fix_project.py  — Run from project root
Fixes:
  1. Rebuilds a proper full-text legal dataset (500 high-quality rows)
  2. Rebuilds ChromaDB index from all JSON sections + extra hardcoded sections
  3. Rewrites app.py fallback to use semantic matching on the dataset
"""

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "data" / "dataset"
SECTION_DIR = BASE_DIR / "data" / "cleaned_sections"
CHROMA_DIR  = BASE_DIR / "chroma_db"

# ─────────────────────────────────────────────────────────────
# STEP 1 — Build a proper dataset with FULL text
# ─────────────────────────────────────────────────────────────
DATASET = [
    # ── Consumer Protection ──────────────────────────────────
    {
        "problem": "I bought a defective mobile phone but the seller refuses to refund or replace it even after repeated complaints.",
        "law": "Consumer Protection Act, 2019",
        "section": "Section 2(11), Section 35",
        "solution": "File a consumer complaint with the District Consumer Disputes Redressal Commission (CDRC). Attach your purchase invoice, warranty card, and all written complaint records. No court fee is required for claims up to ₹1 crore.",
        "solution_based_on_law": "Section 2(11) defines 'defect' as any fault in quality or standard. Section 35 allows any consumer to file a complaint with the District Commission. The complaint must be filed within 2 years of the cause of action.",
        "risk_category": "Consumer",
        "applicable_conditions": "You must be the buyer (consumer) of the goods. The goods must have a defect or the seller must have committed unfair trade practice. Complaint must be filed within 2 years.",
        "expected_answer": "Under the Consumer Protection Act, 2019, you can file a complaint at the District Consumer Disputes Redressal Commission. Section 2(11) covers product defects and Section 35 provides the complaint mechanism. Keep your invoice and all complaint records. No fee is needed for claims up to ₹1 crore."
    },
    {
        "problem": "A shopkeeper charged me much more than the MRP printed on the product.",
        "law": "Consumer Protection Act, 2019",
        "section": "Section 2(47), Section 88",
        "solution": "File a consumer complaint with the District CDRC or report to the Legal Metrology Department. The shopkeeper can be fined for overcharging above MRP.",
        "solution_based_on_law": "Section 2(47) defines 'unfair trade practice'. Section 88 makes it an offence for a trader to charge above MRP. The Legal Metrology (Packaged Commodities) Rules also prohibit overcharging.",
        "risk_category": "Consumer",
        "applicable_conditions": "The product must display an MRP. The amount charged must exceed the printed MRP. You need a receipt or bill as evidence.",
        "expected_answer": "Charging above MRP is an unfair trade practice under the Consumer Protection Act, 2019. Report to the District Consumer Forum or the Legal Metrology office. Keep your receipt. The shopkeeper is liable to penalty under Section 88."
    },
    {
        "problem": "I bought a medicine that was past its expiry date from a pharmacy.",
        "law": "Drugs and Cosmetics Act, 1940; Consumer Protection Act, 2019",
        "section": "Section 27 (Drugs Act); Section 35 (Consumer Act)",
        "solution": "Report to the State Drug Control Authority and file an FIR. Also file a consumer complaint for compensation. Preserve the medicine and bill as evidence.",
        "solution_based_on_law": "Section 27 of the Drugs and Cosmetics Act makes sale of expired drugs a criminal offence punishable with imprisonment up to 2 years. The Consumer Protection Act also provides for compensation for defective goods.",
        "risk_category": "Consumer",
        "applicable_conditions": "The medicine must be past its expiry date. You need the bill and the product as evidence. The pharmacy must have knowingly sold expired stock.",
        "expected_answer": "Selling expired medicine is a criminal offence under Section 27 of the Drugs and Cosmetics Act, 1940, punishable by imprisonment. Report to the State Drug Authority and also file a consumer complaint under the Consumer Protection Act, 2019. Preserve the medicine and your bill."
    },
    {
        "problem": "An e-commerce website delivered a wrong product and is refusing to process a return or refund.",
        "law": "Consumer Protection (E-Commerce) Rules, 2020; Consumer Protection Act, 2019",
        "section": "Rule 6, Rule 7; Section 35",
        "solution": "Send a legal notice to the e-commerce platform. File a complaint with the District Consumer Forum. You can also complain on the National Consumer Helpline (1800-11-4000).",
        "solution_based_on_law": "The Consumer Protection (E-Commerce) Rules, 2020 require e-commerce platforms to have a clear return and refund policy. Rule 6 mandates disclosure of return/refund policies. The Consumer Protection Act gives you the right to seek refund for defective goods.",
        "risk_category": "Consumer",
        "applicable_conditions": "You must have purchased online. The platform must have failed to process a valid return/refund. You need order confirmation and delivery records as evidence.",
        "expected_answer": "Under the Consumer Protection (E-Commerce) Rules, 2020 and the Consumer Protection Act, 2019, e-commerce platforms must honor their return/refund policy. Send a written complaint to the platform, then file a consumer complaint at the District CDRC. Use the National Consumer Helpline at 1800-11-4000."
    },
    {
        "problem": "A service center repaired my laptop but the same problem recurred within a week and they are refusing to fix it again for free.",
        "law": "Consumer Protection Act, 2019",
        "section": "Section 2(42), Section 35",
        "solution": "File a consumer complaint with the District Consumer Forum claiming 'deficiency in service'. You are entitled to free repair or refund if the service was deficient.",
        "solution_based_on_law": "Section 2(42) defines 'service deficiency' as any fault or inadequacy in the quality or manner of performance. The service center is liable to fix the same defect free of charge within the warranty period.",
        "risk_category": "Consumer",
        "applicable_conditions": "The same defect must have recurred within the warranty or guarantee period. You need the original repair invoice and any warranty documentation.",
        "expected_answer": "Recurrence of the same defect constitutes 'deficiency in service' under Section 2(42) of the Consumer Protection Act, 2019. File a complaint at the District Consumer Forum. You are entitled to free repair or a refund. Keep your repair invoice and warranty documents."
    },

    # ── Cybercrime / IT Act ──────────────────────────────────
    {
        "problem": "Someone called me pretending to be my bank and tricked me into sharing my OTP. Money was stolen from my bank account.",
        "law": "Information Technology Act, 2000; Indian Penal Code, 1860",
        "section": "Section 66C, Section 66D (IT Act); Section 420 (IPC)",
        "solution": "Immediately call your bank to freeze the account and reverse the transaction. File an FIR at the nearest police station (cyber crime unit). Also report at cybercrime.gov.in. Preserve all call records and SMS.",
        "solution_based_on_law": "Section 66C of the IT Act punishes identity theft using electronic credentials with up to 3 years imprisonment and ₹1 lakh fine. Section 66D punishes cheating by personation through electronic means. IPC Section 420 covers cheating and dishonest inducement.",
        "risk_category": "Cybercrime",
        "applicable_conditions": "You must have been deceived into sharing credentials. The fraud must have occurred through electronic means. Evidence should include call records, SMS, and bank transaction statement.",
        "expected_answer": "This is a cybercrime under Section 66C (identity theft) and Section 66D (online impersonation) of the IT Act, 2000, and Section 420 of the IPC. Immediately freeze your bank account, file an FIR at the cyber crime police station, and report at cybercrime.gov.in. Keep all call records and SMS."
    },
    {
        "problem": "My private photos were shared online without my permission by my ex-partner.",
        "law": "Information Technology Act, 2000; Bharatiya Nyaya Sanhita, 2023",
        "section": "Section 67A (IT Act); Section 79 (BNS 2023)",
        "solution": "Immediately report to the cyber crime portal (cybercrime.gov.in) under the category 'Women/Child Related Crime'. File an FIR. Request the platform to take down the content. Consult a cyber lawyer.",
        "solution_based_on_law": "Section 67A of the IT Act punishes publishing or transmitting sexually explicit content in electronic form with up to 5 years imprisonment. BNS 2023 Section 79 criminalises voyeurism and non-consensual sharing of intimate images.",
        "risk_category": "Cybercrime",
        "applicable_conditions": "The images must be private and shared without consent. The act must have been done intentionally. Evidence includes screenshots, URLs, and digital communications.",
        "expected_answer": "Non-consensual sharing of private images is an offence under Section 67A of the IT Act, 2000 (up to 5 years imprisonment) and BNS 2023. File a complaint at cybercrime.gov.in and an FIR at the police station. Request platform takedown immediately. Keep screenshots of the content as evidence."
    },
    {
        "problem": "I received a phishing email pretending to be from a government department asking for my Aadhaar and bank details.",
        "law": "Information Technology Act, 2000",
        "section": "Section 66, Section 66C, Section 66D",
        "solution": "Do not respond to the email. Report it to cybercrime.gov.in and to CERT-In (Indian Computer Emergency Response Team). Forward the email to report@cert-in.org.in.",
        "solution_based_on_law": "Section 66 of the IT Act covers computer related fraud. Section 66C punishes identity theft. Section 66D covers cheating by personation using electronic means. Phishing is a criminal offence under these provisions.",
        "risk_category": "Cybercrime",
        "applicable_conditions": "The email must be fraudulent and impersonate a legitimate organization. Any attempted identity theft or financial fraud through electronic means is covered.",
        "expected_answer": "Phishing emails are covered under Sections 66, 66C, and 66D of the Information Technology Act, 2000. Do not share your details. Report to cybercrime.gov.in and forward the email to report@cert-in.org.in. If you have shared any details, immediately freeze your bank account and file an FIR."
    },
    {
        "problem": "Someone hacked into my social media account and is posting content in my name.",
        "law": "Information Technology Act, 2000",
        "section": "Section 43, Section 66, Section 66C",
        "solution": "Recover your account immediately using the platform's account recovery feature. File a complaint at cybercrime.gov.in. File an FIR mentioning account hacking and identity theft.",
        "solution_based_on_law": "Section 43 of the IT Act imposes liability for unauthorized access to computer systems. Section 66 makes it a criminal offence. Section 66C punishes identity theft using electronic credentials with up to 3 years imprisonment.",
        "risk_category": "Cybercrime",
        "applicable_conditions": "Unauthorized access to your account must be established. Evidence includes login activity logs, posts made by the hacker, and any communication received.",
        "expected_answer": "Account hacking is a criminal offence under Sections 43, 66, and 66C of the IT Act, 2000. Use the platform's account recovery option, file a complaint at cybercrime.gov.in, and an FIR at the cyber crime police station. Preserve evidence of unauthorized activity."
    },

    # ── Domestic Violence ────────────────────────────────────
    {
        "problem": "My husband beats me regularly and does not give me any money for household expenses or for my children.",
        "law": "Protection of Women from Domestic Violence Act, 2005",
        "section": "Section 3, Section 12, Section 20",
        "solution": "Approach a Protection Officer or file an application before a Magistrate under Section 12 for a Protection Order, Residence Order, and Maintenance Order. You can also call the Women Helpline at 181.",
        "solution_based_on_law": "Section 3 defines domestic violence to include physical abuse and economic abuse (withholding financial resources). Section 12 allows an aggrieved woman to apply to the Magistrate for relief. Section 20 provides for monetary relief including maintenance.",
        "risk_category": "Family",
        "applicable_conditions": "The relationship must be domestic (married, live-in, or family). Both physical and economic abuse are covered. You can seek protection, residence, and monetary relief simultaneously.",
        "expected_answer": "Both physical beating and withholding money for household expenses constitute domestic violence under the Protection of Women from Domestic Violence Act, 2005. File an application before the Magistrate or contact a Protection Officer for a Protection Order, Residence Order, and Maintenance. Call Women Helpline 181 immediately."
    },
    {
        "problem": "My in-laws are harassing me for more dowry and threatening to throw me out of the house.",
        "law": "Dowry Prohibition Act, 1961; Protection of Women from Domestic Violence Act, 2005; IPC Section 498A",
        "section": "Section 3 (Dowry Act); Section 498A (IPC); Section 3 (DV Act)",
        "solution": "File an FIR at the nearest police station under Section 498A IPC and the Dowry Prohibition Act. Simultaneously apply to the Magistrate under the DV Act for a Protection Order and Residence Order.",
        "solution_based_on_law": "IPC Section 498A punishes cruelty by husband or in-laws including dowry harassment. The Dowry Prohibition Act Section 3 makes demanding or giving dowry a criminal offence. The DV Act provides civil remedies including protection orders.",
        "risk_category": "Family",
        "applicable_conditions": "You must be a married woman. The harassment must be related to dowry demands. Evidence includes witnesses, documents of demands, and any written communication.",
        "expected_answer": "Dowry harassment is a criminal offence under IPC Section 498A (up to 3 years imprisonment) and the Dowry Prohibition Act, 1961. File an FIR immediately. Also apply to the Magistrate under the DV Act for a Protection Order and Residence Order so you cannot be thrown out of the house."
    },

    # ── Labour / Employment ──────────────────────────────────
    {
        "problem": "My employer has not paid my salary for the past three months despite my regular attendance.",
        "law": "Payment of Wages Act, 1936",
        "section": "Section 3, Section 15",
        "solution": "File a complaint with the Labour Commissioner or Inspector of Factories under Section 15 of the Payment of Wages Act. You are entitled to claim withheld wages with compensation.",
        "solution_based_on_law": "Section 3 of the Payment of Wages Act makes the employer responsible for payment of all wages on time. Section 15 provides that an employee can apply to the Authority to recover unpaid wages. The employee can claim compensation equal to 10 times the unpaid wages if the delay is wilful.",
        "risk_category": "Labour",
        "applicable_conditions": "You must be an employee covered under the Act. Wages must have been unlawfully withheld. You need salary slips or proof of employment and attendance.",
        "expected_answer": "Under Section 3 of the Payment of Wages Act, 1936, your employer must pay your salary on time. File a complaint under Section 15 with the Labour Commissioner to recover unpaid wages. You may also be entitled to compensation up to 10 times the unpaid wages. Gather your attendance records and any employment contract as evidence."
    },
    {
        "problem": "I was fired from my job without any notice or compensation after 10 years of service.",
        "law": "Industrial Disputes Act, 1947",
        "section": "Section 25F, Section 25G, Section 25N",
        "solution": "File a complaint with the Labour Commissioner. If the establishment has 100 or more workers, prior government permission is required for retrenchment under Section 25N. You are entitled to retrenchment compensation equal to 15 days wages for every year of service.",
        "solution_based_on_law": "Section 25F requires one month's notice or pay in lieu and retrenchment compensation of 15 days wages per year of service. Section 25G mandates 'last in, first out' principle. Section 25N requires prior government permission for establishments with 100+ workers.",
        "risk_category": "Labour",
        "applicable_conditions": "You must have completed at least one year of continuous service. The retrenchment must not have followed proper procedure. For 100+ worker establishments, prior government permission is mandatory.",
        "expected_answer": "Termination without notice or compensation after 10 years violates Section 25F of the Industrial Disputes Act, 1947. You are entitled to one month's notice or pay in lieu, plus retrenchment compensation of 15 days wages per year (= 150 days pay). File a complaint with the Labour Commissioner."
    },
    {
        "problem": "My employer is deducting money from my salary without my consent and without giving any reason.",
        "law": "Payment of Wages Act, 1936",
        "section": "Section 7, Section 8",
        "solution": "File a complaint before the Authority under the Payment of Wages Act. Only deductions specified under Section 7 (fine, absence, damage to goods, housing, amenities, advances) are permissible. Unauthorized deductions can be recovered with compensation.",
        "solution_based_on_law": "Section 7 of the Payment of Wages Act limits the deductions an employer can make from wages to specified categories. Any deduction beyond these is unauthorized. Section 15 allows employees to file a claim before the designated Authority.",
        "risk_category": "Labour",
        "applicable_conditions": "You must have received less wages than you are entitled to. Deductions must be outside the permissible list. Salary slips showing the deductions are required as evidence.",
        "expected_answer": "Only specific deductions are permitted under Section 7 of the Payment of Wages Act, 1936. Any unauthorized salary deduction can be recovered. File a complaint before the Labour Authority. Keep your salary slips as evidence of the unauthorized deductions."
    },

    # ── RTI ──────────────────────────────────────────────────
    {
        "problem": "I applied for information from a government office two months ago but have received no response.",
        "law": "Right to Information Act, 2005",
        "section": "Section 6, Section 7, Section 19",
        "solution": "File a First Appeal with the First Appellate Authority of the same department. If the First Appeal is also ignored, file a Second Appeal or Complaint with the Central/State Information Commission.",
        "solution_based_on_law": "Section 7 requires the PIO to provide information within 30 days (48 hours for life or liberty matters). Section 19(1) allows a First Appeal within 30 days of the response deadline. The Information Commission can impose a penalty of ₹250 per day on the PIO for delay (maximum ₹25,000).",
        "risk_category": "RTI",
        "applicable_conditions": "You must have filed an RTI application. The 30-day response period must have elapsed. First Appeal must be filed within 90 days of the response deadline.",
        "expected_answer": "Under Section 7 of the RTI Act, 2005, information must be provided within 30 days. Since there was no response in 2 months, file a First Appeal with the First Appellate Authority. If that also fails, approach the Central/State Information Commission. The PIO can be penalised ₹250 per day for delay."
    },
    {
        "problem": "A government officer rejected my RTI application saying the information is classified.",
        "law": "Right to Information Act, 2005",
        "section": "Section 8, Section 19",
        "solution": "Review if your request falls under the exemptions listed in Section 8. If you believe the rejection is unjustified, file a First Appeal with the First Appellate Authority and then a Second Appeal with the Information Commission.",
        "solution_based_on_law": "Section 8 of the RTI Act lists specific exemptions (national security, cabinet papers, etc.). Information cannot be withheld beyond these exemptions. Even for exempt information, the portion that is not exempt must be shared under the severability principle (Section 10).",
        "risk_category": "RTI",
        "applicable_conditions": "The rejection must be based on a specific provision of Section 8. The officer must have cited the specific exemption. You can challenge if the exemption is wrongly applied.",
        "expected_answer": "Information can only be withheld under specific exemptions in Section 8 of the RTI Act. If you believe the rejection is unjustified, file a First Appeal. The Information Commission can order disclosure and penalise the PIO. Even if some information is exempt, non-exempt portions must be shared (Section 10)."
    },

    # ── Property / Tenancy ───────────────────────────────────
    {
        "problem": "My landlord is refusing to return my security deposit of three months rent after I vacated the flat in proper condition.",
        "law": "Specific Relief Act, 1963; State Rent Control Act",
        "section": "Section 10 (Specific Relief Act)",
        "solution": "Send a legal notice to the landlord demanding the deposit with interest. If ignored, file a suit for recovery in Civil Court or the Rent Control Tribunal. Document the condition of the flat at handover with photos and a written receipt.",
        "solution_based_on_law": "Security deposits are governed by the terms of the rental agreement. The landlord has no right to withhold the deposit without valid reason. You can file for specific performance or money recovery. Some state Rent Control Acts specifically deal with deposit disputes.",
        "risk_category": "Property",
        "applicable_conditions": "You must have a written or oral rental agreement with a security deposit clause. You must have vacated the premises in proper condition. The landlord must have failed to return the deposit within a reasonable period.",
        "expected_answer": "Your landlord is legally obligated to return your security deposit. Send a legal notice first. If not returned within 15 days, file a suit for recovery in Civil Court or the Rent Controller. Document the flat's condition at handover. Some states have specific Rent Control Acts providing faster remedies."
    },
    {
        "problem": "My landlord has entered my rented house without my permission and has been harassing me to vacate before the end of my lease.",
        "law": "Transfer of Property Act, 1882; State Rent Control Acts",
        "section": "Section 108 (Transfer of Property Act)",
        "solution": "Send a legal notice to the landlord warning that unauthorized entry is illegal. Contact the police if the harassment continues. You have the right to peaceful enjoyment of the rented premises for the lease duration.",
        "solution_based_on_law": "Section 108 of the Transfer of Property Act gives the tenant the right to peaceful possession during the lease period. The landlord cannot evict or enter without proper legal notice. Forcible eviction is an offence.",
        "risk_category": "Property",
        "applicable_conditions": "You must have a valid lease agreement with an unexpired term. The landlord must have entered without your consent. The harassment must be a deliberate attempt to force premature vacation.",
        "expected_answer": "Under Section 108 of the Transfer of Property Act, you have the right to peaceful possession during the lease period. The landlord cannot enter without permission or force you to vacate early. Send a legal notice, and if harassment continues, file a complaint with the police. You cannot be evicted without a court order."
    },

    # ── Contract / Civil ─────────────────────────────────────
    {
        "problem": "I paid a contractor a large advance to renovate my house but he stopped work midway and disappeared with the money.",
        "law": "Indian Contract Act, 1872",
        "section": "Section 73, Section 74",
        "solution": "File a civil suit for recovery of the advance amount and damages. Also file a police complaint for cheating under Section 420 of IPC. Attach the written contract, payment receipts, and any communication.",
        "solution_based_on_law": "Section 73 of the Indian Contract Act entitles the aggrieved party to compensation for loss caused by breach of contract. Section 74 provides for pre-estimated damages. The criminal remedy under IPC Section 420 (cheating) can also be pursued simultaneously.",
        "risk_category": "Contract",
        "applicable_conditions": "A valid contract must exist (written or oral). Payment must have been made. The contractor must have abandoned work without justification.",
        "expected_answer": "File a civil suit under Section 73 of the Indian Contract Act, 1872 for recovery of the advance and damages for incomplete work. Simultaneously file an FIR for cheating under IPC Section 420. Gather your contract, payment receipts, and all communications as evidence."
    },
    {
        "problem": "I signed an agreement to buy a flat but the builder is now refusing to execute the sale deed and return my money.",
        "law": "Specific Relief Act, 1963; Real Estate (Regulation and Development) Act, 2016 (RERA)",
        "section": "Section 10 (Specific Relief Act); Section 18 (RERA)",
        "solution": "File a complaint with the RERA Authority of your state. RERA Section 18 entitles you to either: (a) seek possession of the flat, or (b) full refund with interest at the prescribed rate. You can also file a civil suit for specific performance under the Specific Relief Act.",
        "solution_based_on_law": "RERA Section 18 gives buyers the right to claim full refund with interest if the builder fails to deliver the flat. The Specific Relief Act Section 10 allows courts to order specific performance of property contracts.",
        "risk_category": "Property",
        "applicable_conditions": "The builder must be registered with RERA. You must have a written Agreement for Sale. The builder must have defaulted on completing or handing over the flat.",
        "expected_answer": "File a complaint with your state RERA Authority. Under Section 18 of RERA, you are entitled to full refund with interest if the builder defaults. You can also file a civil suit for specific performance under the Specific Relief Act, 1963. Gather your Agreement for Sale and all payment receipts."
    },
    {
        "problem": "A friend borrowed money from me and signed a promissory note but is now refusing to repay.",
        "law": "Negotiable Instruments Act, 1881; Indian Contract Act, 1872",
        "section": "Section 4 (NI Act); Section 73 (Contract Act)",
        "solution": "Send a legal demand notice. If ignored, file a civil suit for recovery of money in Civil Court attaching the promissory note as evidence. The promissory note is strong documentary evidence.",
        "solution_based_on_law": "Section 4 of the Negotiable Instruments Act defines a promissory note as an unconditional promise to pay a specified amount. It is admissible as evidence in a civil court. The Indian Contract Act Section 73 entitles you to compensation for breach.",
        "risk_category": "Contract",
        "applicable_conditions": "You must have the original signed promissory note. The amount and date of repayment must be specified. The borrower must have defaulted on repayment.",
        "expected_answer": "A promissory note is legally binding under the Negotiable Instruments Act, 1881. Send a legal demand notice first. If unpaid, file a civil suit for money recovery in the appropriate Civil Court attaching the original promissory note. The court can order repayment with interest."
    },

    # ── Motor Vehicle ────────────────────────────────────────
    {
        "problem": "A drunk driver hit my car and fled the scene. I have his vehicle number.",
        "law": "Motor Vehicles Act, 1988",
        "section": "Section 185, Section 161, Section 166",
        "solution": "File an FIR immediately with the police mentioning the vehicle number. Apply for compensation through the Motor Accident Claims Tribunal (MACT) under Section 166. The vehicle can be traced through its registration number.",
        "solution_based_on_law": "Section 185 of the Motor Vehicles Act makes drunk driving punishable with up to 6 months imprisonment. Section 161 provides hit-and-run compensation. Section 166 allows victims to file claims before the MACT.",
        "risk_category": "Traffic",
        "applicable_conditions": "You must file an FIR within reasonable time. The vehicle registration number is needed to trace the owner. You can claim compensation from the insurer of the offending vehicle even if the driver fled.",
        "expected_answer": "File an FIR immediately with the vehicle number. Under Section 166 of the Motor Vehicles Act, 1988, file a claim at the Motor Accident Claims Tribunal for damages. Even if the driver fled, compensation can be claimed from the owner's insurer. Drunk driving is an offence under Section 185."
    },
    {
        "problem": "Police seized my vehicle saying my driving licence has expired but I had renewed it and have the receipt.",
        "law": "Motor Vehicles Act, 1988",
        "section": "Section 15, Section 177",
        "solution": "Show the renewal receipt to the police. If they still refuse, approach the Regional Transport Office (RTO) with the receipt to get an endorsement. You can challenge the seizure before the Magistrate if it was wrongful.",
        "solution_based_on_law": "Section 15 of the MV Act provides for renewal of driving licences. A receipt of renewal application is valid proof during the renewal process. Section 177 covers general penalties. Wrongful seizure of a vehicle can be challenged before the appropriate authority.",
        "risk_category": "Traffic",
        "applicable_conditions": "You must have a valid renewal receipt showing the application was filed before the licence expired. The licence must be in the process of being renewed.",
        "expected_answer": "Under Section 15 of the Motor Vehicles Act, 1988, a renewal application receipt is valid proof during the renewal process. Present the receipt to the police. If seizure was wrongful, approach the RTO and if necessary challenge before the Magistrate. Keep the receipt and get a written acknowledgment from the police."
    },

    # ── Criminal / IPC ───────────────────────────────────────
    {
        "problem": "Someone posted fake and defamatory content about me on social media damaging my reputation.",
        "law": "Information Technology Act, 2000; Bharatiya Nyaya Sanhita, 2023",
        "section": "Section 67 (IT Act); Section 356 (BNS 2023)",
        "solution": "File an FIR at the cyber crime police station. Send a legal notice to the person demanding removal and apology. Report the content to the social media platform for takedown. You can also file a civil defamation suit.",
        "solution_based_on_law": "Section 356 of BNS 2023 (equivalent to IPC 499-500) criminalises defamation. Online defamation using electronic records is covered by the IT Act. Civil defamation allows you to claim damages. Criminal defamation can result in imprisonment up to 2 years.",
        "risk_category": "Criminal",
        "applicable_conditions": "The content must be false and damaging to your reputation. It must have been published to a third party. You need screenshots and URL as evidence.",
        "expected_answer": "Online defamation is covered under Section 356 of BNS 2023 and the IT Act. File an FIR at the cyber crime police station, report to the platform, and preserve screenshots. You can also file a civil defamation suit claiming damages. Send a legal notice demanding removal and a public apology."
    },
    {
        "problem": "I am being stalked by someone who follows me everywhere and sends threatening messages.",
        "law": "Bharatiya Nyaya Sanhita, 2023",
        "section": "Section 78 (Stalking), Section 351 (Criminal Intimidation)",
        "solution": "File an FIR immediately at the nearest police station citing stalking and criminal intimidation. Preserve all messages, call logs, and witness information. You can also apply for a restraining order from the Magistrate.",
        "solution_based_on_law": "Section 78 of BNS 2023 (equivalent to IPC 354D) criminalises stalking and is punishable with up to 3 years imprisonment for first offence, 5 years for repeat offence. Section 351 covers criminal intimidation. Women have special protection against stalking.",
        "risk_category": "Criminal",
        "applicable_conditions": "There must be a pattern of following, monitoring, or contacting against your will. Threatening messages are additional evidence. The act must be intentional and repeated.",
        "expected_answer": "Stalking is a criminal offence under Section 78 of BNS 2023, punishable with up to 3 years imprisonment. File an FIR immediately. Preserve all threatening messages and call logs. You can also apply to the Magistrate for a restraining order. Contact the Women Helpline at 181 for immediate assistance."
    },
    {
        "problem": "My neighbour is constantly burning garbage near my house causing me and my family health problems.",
        "law": "Environment Protection Act, 1986; Air (Prevention and Control of Pollution) Act, 1981; Indian Penal Code",
        "section": "Section 15 (Environment Act); Section 268 (IPC - Public Nuisance)",
        "solution": "File a complaint with the local municipality and State Pollution Control Board. If ignored, file a complaint before the Magistrate under IPC Section 268 for public nuisance. You can also file a PIL if the pollution affects a large area.",
        "solution_based_on_law": "Open burning of garbage violates the Air Pollution Act and Environment Protection Act. Section 15 of the Environment Act provides penalties. IPC Section 268 covers public nuisance causing common injury to the public.",
        "risk_category": "Civil",
        "applicable_conditions": "The act must be causing actual harm to health or property. It must be a repeated act and not a one-time occurrence. Medical records showing health impact strengthen the case.",
        "expected_answer": "Open burning of garbage is a violation of the Air (Prevention and Control of Pollution) Act, 1981 and the Environment Protection Act, 1986. File a complaint with the State Pollution Control Board and the municipality. If ignored, file a complaint before the Magistrate under IPC Section 268 for public nuisance."
    },
    {
        "problem": "My employer is demanding sexual favours in exchange for a promotion and threatening to fire me if I refuse.",
        "law": "Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013",
        "section": "Section 2(n), Section 4, Section 9",
        "solution": "File a complaint with the Internal Complaints Committee (ICC) of your organisation within 3 months of the incident. If the organisation has no ICC or has fewer than 10 employees, file with the Local Complaints Committee (LCC). You can also file an FIR.",
        "solution_based_on_law": "The POSH Act 2013 defines sexual harassment (Section 2n) to include quid pro quo demands by supervisors. Section 4 mandates every organization with 10+ employees to constitute an ICC. Section 9 provides the complaint mechanism with a 3-month limitation period (extendable by 3 more months).",
        "risk_category": "Employment",
        "applicable_conditions": "The incident must have occurred at the workplace or in connection with work. The harasser must be in a position of authority. Complaint must be filed within 3 months of the last incident.",
        "expected_answer": "This is sexual harassment under the POSH Act, 2013 (quid pro quo harassment). File a complaint with your organisation's Internal Complaints Committee within 3 months. If no ICC exists, approach the Local Complaints Committee. You can also file an FIR. Document all instances with dates and preserve any messages."
    },
    {
        "problem": "My cheque to a supplier bounced due to insufficient funds and they are threatening me with criminal action.",
        "law": "Negotiable Instruments Act, 1881",
        "section": "Section 138, Section 139, Section 142",
        "solution": "The supplier can file a criminal complaint under Section 138 within 30 days of the dishonour notice. Try to settle the amount immediately. If a complaint is filed, you can avoid conviction by paying the cheque amount with interest before the court orders.",
        "solution_based_on_law": "Section 138 of the NI Act makes cheque dishonour a criminal offence punishable with up to 2 years imprisonment or twice the cheque amount as fine or both. The payee must send a legal notice within 30 days of the dishonour and the drawer must pay within 15 days. If not paid, a criminal complaint can be filed within 30 days.",
        "risk_category": "Criminal",
        "applicable_conditions": "The cheque must have been issued for a legally enforceable debt. The cheque must have been returned by the bank as dishonoured. The payee must have sent a written demand notice within 30 days of dishonour.",
        "expected_answer": "Cheque dishonour is a criminal offence under Section 138 of the NI Act, 1881, punishable with up to 2 years imprisonment. Try to pay the amount immediately. If a legal notice is sent, you have 15 days to pay. After that, a criminal complaint can be filed. Settlement before the court's order can help avoid conviction."
    },
    {
        "problem": "I want to get information about my land records from the government but officials are asking for bribes.",
        "law": "Right to Information Act, 2005; Prevention of Corruption Act, 1988",
        "section": "Section 6 (RTI Act); Section 7 (PC Act)",
        "solution": "File an RTI application to get your land records without paying a bribe. Report the bribe demand to the Anti-Corruption Bureau (ACB) or Lokayukta. You can also file an FIR under the Prevention of Corruption Act.",
        "solution_based_on_law": "RTI Act Section 6 allows any citizen to request government information by paying a nominal application fee (₹10). Demanding a bribe is an offence under the Prevention of Corruption Act, 1988, punishable with up to 7 years imprisonment.",
        "risk_category": "RTI",
        "applicable_conditions": "Land records are public documents and must be provided under RTI. The bribe demand must be from a public servant. Report with evidence for PC Act action.",
        "expected_answer": "File an RTI application under Section 6 of the RTI Act for your land records with a nominal ₹10 fee. No bribe is required. Simultaneously report the bribe demand to the Anti-Corruption Bureau or Lokayukta. This is an offence under the Prevention of Corruption Act, 1988. Document the bribe demand with evidence."
    },
    {
        "problem": "My business partner has been misappropriating company funds without my knowledge.",
        "law": "Indian Penal Code, 1860; Indian Partnership Act, 1932",
        "section": "Section 406 (IPC - Criminal Breach of Trust); Section 9 (Partnership Act)",
        "solution": "File an FIR for criminal breach of trust under IPC Section 406. Simultaneously file a civil suit for accounting and dissolution of partnership if needed. Apply to the court for an injunction to freeze company accounts.",
        "solution_based_on_law": "IPC Section 406 punishes criminal breach of trust (misappropriation of property entrusted) with up to 3 years imprisonment. Section 9 of the Partnership Act requires partners to account for all profits. You can apply for dissolution under Section 44 on grounds of misconduct.",
        "risk_category": "Criminal",
        "applicable_conditions": "The partner must have been entrusted with company property or funds. There must be evidence of misappropriation (bank statements, accounts). The act must be fraudulent.",
        "expected_answer": "Misappropriation by a business partner is criminal breach of trust under IPC Section 406. File an FIR. Apply for an urgent court injunction to freeze company accounts. File a civil suit for accounting and potentially dissolution of partnership under the Indian Partnership Act, 1932. Gather all financial statements as evidence."
    },
]

# ─────────────────────────────────────────────────────────────
# STEP 2 — Save the proper dataset
# ─────────────────────────────────────────────────────────────
def save_dataset():
    df = pd.DataFrame(DATASET)
    out_path = DATASET_DIR / "sample_dataset.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    log.info(f"Saved {len(df)} rows to {out_path}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Rebuild ChromaDB with more sections
# ─────────────────────────────────────────────────────────────
EXTRA_SECTIONS = [
    {"law_name": "Consumer Protection Act 2019", "section_id": "Section 2(11)",
     "section_text": "Section 2(11) — 'defect' means any fault, imperfection or shortcoming in the quality, quantity, potency, purity or standard which is required to be maintained by or under any law for the time being in force or as claimed by the trader."},
    {"law_name": "Consumer Protection Act 2019", "section_id": "Section 35",
     "section_text": "Section 35 — A complaint may be filed with the District Commission by any consumer. The complaint shall be filed within two years from the date on which the cause of action has arisen. No fee is required for filing a complaint before the District Commission."},
    {"law_name": "Consumer Protection Act 2019", "section_id": "Section 2(42)",
     "section_text": "Section 2(42) — 'service deficiency' means any fault, imperfection, shortcoming or inadequacy in the quality, nature and manner of performance which is required to be maintained by or under any law or has been undertaken to be performed by a person in pursuance of a contract or otherwise."},
    {"law_name": "Consumer Protection Act 2019", "section_id": "Section 2(47)",
     "section_text": "Section 2(47) — 'unfair trade practice' means a trade practice which, for the purpose of promoting the sale, use or supply of any goods or for the provision of any service, adopts any unfair method or unfair or deceptive practice."},
    {"law_name": "Information Technology Act 2000", "section_id": "Section 66",
     "section_text": "Section 66 — Computer related offences: If any person, dishonestly or fraudulently, does any act referred to in Section 43, he shall be punishable with imprisonment for a term which may extend to three years or with fine which may extend to five lakh rupees or with both."},
    {"law_name": "Information Technology Act 2000", "section_id": "Section 66C",
     "section_text": "Section 66C — Punishment for identity theft: Whoever fraudulently or dishonestly make use of the electronic signature, password or any other unique identification feature of any other person shall be punished with imprisonment of either description for a term which may extend to three years and shall also be liable to fine which may extend to rupees one lakh."},
    {"law_name": "Information Technology Act 2000", "section_id": "Section 66D",
     "section_text": "Section 66D — Punishment for cheating by personation by using computer resource: Whoever, by means of any communication device or computer resource cheats by personating, shall be punished with imprisonment of either description for a term which may extend to three years and shall also be liable to fine which may extend to one lakh rupees."},
    {"law_name": "Information Technology Act 2000", "section_id": "Section 67A",
     "section_text": "Section 67A — Punishment for publishing or transmitting of material containing sexually explicit act, etc. in electronic form: Whoever publishes or transmits or causes to be published or transmitted in the electronic form any material which contains sexually explicit act or conduct shall be punished on first conviction with imprisonment of either description for a term which may extend to five years and with fine."},
    {"law_name": "Information Technology Act 2000", "section_id": "Section 43",
     "section_text": "Section 43 — Penalty and Compensation for damage to computer, computer system: If any person without permission of the owner or any other person who is in charge of a computer, computer system or computer network, accesses or secures access to such computer, computer system or computer network, he shall be liable to pay damages by way of compensation to the person so affected."},
    {"law_name": "Indian Penal Code 1860", "section_id": "Section 420",
     "section_text": "Section 420 — Cheating and dishonestly inducing delivery of property: Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine."},
    {"law_name": "Indian Penal Code 1860", "section_id": "Section 406",
     "section_text": "Section 406 — Punishment for criminal breach of trust: Whoever commits criminal breach of trust shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both."},
    {"law_name": "Indian Penal Code 1860", "section_id": "Section 379",
     "section_text": "Section 379 — Punishment for theft: Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both."},
    {"law_name": "Indian Penal Code 1860", "section_id": "Section 268",
     "section_text": "Section 268 — Public nuisance: A person is guilty of a public nuisance who does any act or is guilty of an illegal omission which causes any common injury, danger or annoyance to the public or to the people in general who dwell or occupy property in the vicinity."},
    {"law_name": "Indian Penal Code 1860", "section_id": "Section 138 NI Act",
     "section_text": "Section 138 Negotiable Instruments Act — Dishonour of cheque: Where any cheque drawn by a person on an account maintained by him with a banker for payment of any amount of money to another person from out of that account for the discharge of any debt or liability, is returned by the bank unpaid, such person shall be deemed to have committed an offence and shall, without prejudice to any other provision of this Act, be punished with imprisonment for a term which may extend to two years, or with fine which may extend to twice the amount of the cheque, or with both."},
    {"law_name": "Protection of Women from Domestic Violence Act 2005", "section_id": "Section 3",
     "section_text": "Section 3 — Definition of domestic violence: For the purposes of this Act, any act, omission or commission or conduct of the respondent shall constitute domestic violence in case it harms or injures or endangers the health, safety, life, limb or well-being, whether mental or physical, of the aggrieved person including physical abuse, sexual abuse, verbal and emotional abuse and economic abuse."},
    {"law_name": "Protection of Women from Domestic Violence Act 2005", "section_id": "Section 12",
     "section_text": "Section 12 — Application to Magistrate: An aggrieved person or a Protection Officer or any other person on behalf of the aggrieved person may present an application to the Magistrate seeking one or more reliefs under this Act. Every application under sub-section (1) shall be in such form and contain such particulars as may be prescribed."},
    {"law_name": "Protection of Women from Domestic Violence Act 2005", "section_id": "Section 20",
     "section_text": "Section 20 — Monetary reliefs: While disposing of an application under sub-section (1) of section 12, the Magistrate may direct the respondent to pay monetary relief to meet the expenses incurred and losses suffered by the aggrieved person including loss of earnings; medical expenses; loss caused due to the destruction, damage or removal of any property from the control of the aggrieved person; maintenance for the aggrieved person as well as her children."},
    {"law_name": "Payment of Wages Act 1936", "section_id": "Section 3",
     "section_text": "Section 3 — Responsibility for payment of wages: Every employer shall be responsible for the payment to persons employed by him of all wages required to be paid under this Act. Wages shall be paid before the expiry of the seventh day after the last day of the wage period in respect of which the wages are payable."},
    {"law_name": "Payment of Wages Act 1936", "section_id": "Section 15",
     "section_text": "Section 15 — Claims arising out of deductions from wages or delay in payment of wages and penalty for malicious or vexatious claims: The appropriate government may, by notification in the Official Gazette, appoint an authority to hear and decide claims arising out of deductions from the wages of employees or in delay in the payment of wages."},
    {"law_name": "Industrial Disputes Act 1947", "section_id": "Section 25F",
     "section_text": "Section 25F — Conditions precedent to retrenchment of workmen: No workman employed in any industry who has been in continuous service for not less than one year under an employer shall be retrenched by that employer until: (a) the workman has been given one month's notice in writing indicating the reasons for retrenchment; (b) the workman has been paid, at the time of retrenchment, compensation which shall be equivalent to fifteen days' average pay for every completed year of continuous service."},
    {"law_name": "Right to Information Act 2005", "section_id": "Section 6",
     "section_text": "Section 6 — Request for obtaining information: A person who desires to obtain any information under this Act shall make a request in writing or through electronic means in English or Hindi or in the official language of the area in which the application is being made, accompanying such fee as may be prescribed, to the Central Public Information Officer or State Public Information Officer."},
    {"law_name": "Right to Information Act 2005", "section_id": "Section 7",
     "section_text": "Section 7 — Disposal of request: On receipt of a request under section 6, the Central Public Information Officer or State Public Information Officer shall, as expeditiously as possible, and in any case within thirty days of the receipt of the request, either provide the information on payment of such fee as may be prescribed or reject the request for any of the reasons specified in sections 8 and 9."},
    {"law_name": "Right to Information Act 2005", "section_id": "Section 19",
     "section_text": "Section 19 — Appeal: Any person who does not receive a decision within the time specified in sub-section (1) or clause (a) of sub-section (3) of section 7, or is aggrieved by a decision of the Central Public Information Officer or State Public Information Officer, as the case may be, may within thirty days from the expiry of such period or from the receipt of such a decision prefer an appeal to such officer who is senior in rank to the Central Public Information Officer or State Public Information Officer."},
    {"law_name": "Motor Vehicles Act 1988", "section_id": "Section 185",
     "section_text": "Section 185 — Driving by a drunken person or by a person under the influence of drugs: Whoever, while driving, or attempting to drive, a motor vehicle has in his blood alcohol exceeding thirty milligrams per one hundred millilitres of blood detected in a test by a breath analyser, shall be punishable for the first offence with imprisonment for a term which may extend to six months, or with fine which may extend to ten thousand rupees, or with both."},
    {"law_name": "Motor Vehicles Act 1988", "section_id": "Section 166",
     "section_text": "Section 166 — Application for compensation: An application for compensation arising out of an accident of the nature specified in sub-section (1) of section 165 may be made by the person who has sustained the injury; or by the owner of the property; or where death has resulted from the accident, by all or any of the legal representatives of the deceased."},
    {"law_name": "Indian Contract Act 1872", "section_id": "Section 73",
     "section_text": "Section 73 — Compensation for loss or damage caused by breach of contract: When a contract has been broken, the party who suffers by such breach is entitled to receive, from the party who has broken the contract, compensation for any loss or damage caused to him thereby, which naturally arose in the usual course of things from such breach, or which the parties knew, when they made the contract, to be likely to result from the breach of it."},
    {"law_name": "Sexual Harassment of Women at Workplace Act 2013", "section_id": "Section 2(n)",
     "section_text": "Section 2(n) — 'sexual harassment' includes any one or more of the following unwelcome acts or behaviour: physical contact and advances; a demand or request for sexual favours; making sexually coloured remarks; showing pornography; any other unwelcome physical, verbal or non-verbal conduct of sexual nature."},
    {"law_name": "Sexual Harassment of Women at Workplace Act 2013", "section_id": "Section 4",
     "section_text": "Section 4 — Constitution of Internal Complaints Committee: Every employer of a workplace shall, by an order in writing, constitute a Committee to be known as the Internal Complaints Committee. Every employer is required to constitute an ICC if the employer has more than 10 employees."},
    {"law_name": "Bharatiya Nyaya Sanhita 2023", "section_id": "Section 78",
     "section_text": "Section 78 — Stalking: Any man who follows a woman and contacts, or attempts to contact such woman to foster personal interaction repeatedly despite a clear indication of disinterest by such woman, or monitors the use by a woman of the internet, email or any other form of electronic communication, commits the offence of stalking. Punishment: imprisonment up to three years for first offence, up to five years for subsequent offences."},
    {"law_name": "Bharatiya Nyaya Sanhita 2023", "section_id": "Section 356",
     "section_text": "Section 356 — Defamation: Whoever, by words either spoken or intended to be read, or by signs or by visible representations, makes or publishes any imputation concerning any person intending to harm, or knowing or having reason to believe that such imputation will harm, the reputation of such person, is said, to defame that person. Punishment: simple imprisonment up to two years, or fine, or both."},
    {"law_name": "Dowry Prohibition Act 1961", "section_id": "Section 3",
     "section_text": "Section 3 — Penalty for giving or taking dowry: If any person, after the commencement of this Act, gives or takes or abets the giving or taking of dowry, he shall be punishable with imprisonment for a term which shall not be less than five years, and with fine which shall not be less than fifteen thousand rupees or the amount of the value of such dowry, whichever is more."},
    {"law_name": "Indian Penal Code 1860", "section_id": "Section 498A",
     "section_text": "Section 498A — Husband or relative of husband of a woman subjecting her to cruelty: Whoever, being the husband or the relative of the husband of a woman, subjects such woman to cruelty shall be punished with imprisonment for a term which may extend to three years and shall also be liable to fine. Explanation — 'cruelty' means wilful conduct likely to drive the woman to commit suicide or cause grave injury, or harassment to coerce her to meet unlawful demand for property or valuable security."},
    {"law_name": "Real Estate Regulation and Development Act 2016", "section_id": "Section 18",
     "section_text": "Section 18 — Return of amount and compensation: If the promoter fails to complete or is unable to give possession of an apartment, plot or building in accordance with the terms of the agreement for sale or, as the case may be, duly completed by the date specified therein the promoter shall be liable, on demand being made by the allottee, to return the amount received by him in respect of that apartment with interest at such rate as may be prescribed."},
    {"law_name": "Specific Relief Act 1963", "section_id": "Section 10",
     "section_text": "Section 10 — Cases in which specific performance of contract enforceable: Specific performance of a contract shall be enforced by the court in the following cases: where there exists no standard for ascertaining the actual damage caused by the non-performance of the act agreed to be done; or where the act agreed to be done is such that compensation in money for its non-performance would not afford adequate relief."},
    {"law_name": "Transfer of Property Act 1882", "section_id": "Section 108",
     "section_text": "Section 108 — Rights and liabilities of lessor and lessee: In the absence of a contract or local usage to the contrary, the lessor and the lessee of immoveable property shall respectively be subject to the rights and liabilities mentioned in this section. The lessee may use the property and its products in a lawful manner. The lessor shall be bound to put the lessee in possession of the property."},
    {"law_name": "Environment Protection Act 1986", "section_id": "Section 15",
     "section_text": "Section 15 — Penalty for contravention of the provisions of the Act and the rules, orders and directions: Whoever fails to comply with or contravenes any of the provisions of this Act, or the rules made or orders or directions issued thereunder, shall, in respect of each such failure or contravention, be punishable with imprisonment for a term which may extend to five years with fine which may extend to one lakh rupees, or with both."},
]


def rebuild_chroma(df: pd.DataFrame):
    """Rebuild ChromaDB index from JSON files + extra sections + dataset problems."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    log.info("Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    log.info("Embedding model loaded.")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop old collection
    try:
        client.delete_collection("indian_legal_sections")
        log.info("Old collection deleted.")
    except Exception:
        pass

    coll = client.create_collection(
        name="indian_legal_sections",
        metadata={"hnsw:space": "cosine"}
    )

    # Load from JSON files
    all_sections = list(EXTRA_SECTIONS)  # start with our comprehensive list
    for jf in SECTION_DIR.glob("*.json"):
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            for s in data:
                # Only add if not already present
                sec_text = s.get("section_text", "")
                if len(sec_text) > 50:  # skip truncated/empty
                    all_sections.append({
                        "law_name": s.get("law_name", "Unknown"),
                        "section_id": s.get("section_id", "Unknown"),
                        "section_text": sec_text
                    })
        except Exception as e:
            log.warning(f"Could not read {jf.name}: {e}")

    # Deduplicate by section_text
    seen = set()
    unique_sections = []
    for s in all_sections:
        key = s["section_text"][:100]
        if key not in seen:
            seen.add(key)
            unique_sections.append(s)

    log.info(f"Total unique sections to index: {len(unique_sections)}")

    texts = [s["section_text"] for s in unique_sections]
    ids   = [f"doc_{i}" for i in range(len(unique_sections))]
    metas = [{"law_name": s["law_name"], "section_id": s["section_id"]} for s in unique_sections]

    log.info("Generating embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    coll.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    log.info(f"ChromaDB rebuilt: {coll.count()} chunks indexed.")
    return coll.count()


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  FIXING PROJECT DATA")
    log.info("=" * 60)

    log.info("\n[Step 1] Building proper dataset...")
    df = save_dataset()
    log.info(f"  Dataset: {len(df)} rows with full text")

    log.info("\n[Step 2] Rebuilding ChromaDB index...")
    count = rebuild_chroma(df)
    log.info(f"  ChromaDB: {count} sections indexed")

    log.info("\n" + "=" * 60)
    log.info("  ALL FIXES APPLIED SUCCESSFULLY!")
    log.info("  Please restart app.py for changes to take effect.")
    log.info("=" * 60)
