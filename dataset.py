dataset = [
  {
    "pages": [
      "NOVALUX SERVICES LLC — internal memo v3\nrev 03/16/2024 | cycle Q1 | ticket 004271\nnotes: last spring round 03-14-2024; prev annual 2019-03-17; audit 15 Mar 2024\nthread list: Liam Chen; Maria Gomez; Henry Stone; Noah Patel; Andrea Ruiz; David Roe\nrooftop set: AC-7C, AC-9F; site: Northwind Offices; zone: PINE-WALK\nroster excerpt: Ruiz, Andrea; Patel, Noah; Chen, Olivia; Long, Peter; Gomez, Maria\nline items:\n- 7C belts swapped 03/15/24\n- 9F filters swapped 03/14/24\n- motor check 2023-03-17\nnames mentioned elsewhere: Omar Diaz, Jason Miller, Karen Bell, Raj Patel\n--- mid-block ------------------------------------------------\nroute grid R2: Chen, Liam | Gomez, Maria | Stone, Emily | Roe, David | Patel, Noah\ndispatch key [core-record: 2024-03-15] ref A1-7C\nnorth list crossref: Henry Stone; Andrea Ruiz\n----------------------------------------------------------------\nmisc dates: 03/15/2024 09:40; 2024/03/15; 15-03-2024\nfooter tag id NLX-INT-004271\n"
    ],
    "date_rule": "Extract the ISO date appearing between the exact marker '[core-record: ' and the closing ']'.",
    "expected_output": {
      "document_main_date": "2024-03-15",
      "client_information": {
        "first_name": "Emily",
        "last_name": "Stone"
      }
    }
  },
  {
    "pages": [
      "ORION INSURANCE COMPANY — coverage packet\nprint 2025/01/02; cycle close 2024-12-29; sample form rev 12-2024\nreference codes: AX-77-193; BQ-11-004\nnames in circulation: Ethan Li, Priya Nair, Omar Diaz, Laurel Kim, Julia Park\npage markers: P1/2\nlist A (mail sweep): Mendes, Ana; Park, Julia; Li, Ethan; Singh, Maya; Patel, Raj\n--- center band ----------------------------------------------\nwindow text row: Mendes, Carlos | addr token G-441B | @when=2025-02-01@\n--------------------------------------------------------------\nreminders: renewal cycle hits 2026-02-01; example date 02/01/2025 (US)\ntrailing mentions: Raj Patel; Maya Singh; Laurel Kim\n"
    ],
    "date_rule": "Select the YYYY-MM-DD date enclosed by the markers '@when=' and '@'.",
    "expected_output": {
      "document_main_date": "2025-02-01",
      "client_information": {
        "first_name": "Carlos",
        "last_name": "Mendes"
      }
    }
  },
  {
    "pages": [
      "NORTH RIVER BANK — consolidated lines\nhdr date 11/05/2023; period 10/01/23–10/31/23; ref NRB-STAT-1180\nledger notes: auditor Olivia Chen 02-Nov-2023; prior msg 2023-10-29; contact Jason Miller\nmasked acct **** 4831 | manager Tomas Rivera | approver Ellen Wu\npeople mentioned: Peter Sand; Daniel Cho; Cathy Nguyen; Henry Stone\nflow:\n- ACH in 10/30/23\n- fee waive 11-02-2023\n--- center ledger --------------------------------------------\nparticipants: Sand, Peter | Nair, Priya | Chen, Olivia | Miller, Jason\nseal <stamp 2023-11-01> batch 44A\n--------------------------------------------------------------\nfooter fragments: 05-11-2023; Nov 1, 2023; 2023/11/01\n"
    ],
    "date_rule": "Use the date contained between '<stamp ' and '>' exactly as YYYY-MM-DD.",
    "expected_output": {
      "document_main_date": "2023-11-01",
      "client_information": {
        "first_name": "Priya",
        "last_name": "Nair"
      }
    }
  },
  {
    "pages": [
      "OAK CREST PROPERTY MGMT. — unit packet\nbldg: Lakeview Rd., unit 5A; intercom map rev 08/23; parking memo 08-23-2022\nnames across building: Amy Tran; Daniel Ortiz; Raj Patel; Sarah Onu; Michael Lin\nstack A roster: Blake, Sarah (prev); Grant, Oliver (current); Ruiz, Andrea; Gomez, Maria\npage 1/3\n--- carryover data -------------------------------------------\nmailbox panel: 5A GRANT O | 5B TRAN A | 6C ORTIZ D | 3D PATEL R\ninspection mentions: 08/22/2022; utilities 08/25/2022; move target 09/01/2022\n--------------------------------------------------------------\nnext page\n",
      "OAK CREST PROPERTY MGMT. — notes\nvisitors seen: Andrea Ruiz; Maria Gomez; David Roe; Jason Miller; Olivia Chen\nrandom dates: 20-08-2022; 2022/08/20; 08/20/22\n--- mid-strip -------------------------------------------------\nkey timeline |dt|2022-08-20|dt| for archive tag LC-5A\n--------------------------------------------------------------\nother: parking review 2022-08-23; form edits 08-18-2022\npage 2/3\n",
      "OAK CREST PROPERTY MGMT. — misc\nunit map checksum 5A-7F-2C; contact index L.Park; badge review 2022-08-21\nfooter copy ids: OCP-5A-AG\npage 3/3\n"
    ],
    "date_rule": "Extract the date between the exact tokens '|dt|' and '|dt|' on page 2.",
    "expected_output": {
      "document_main_date": "2022-08-20",
      "client_information": {
        "first_name": "Oliver",
        "last_name": "Grant"
      }
    }
  },
  {
    "pages": [
      "CITYCARE CLINIC — visit archive\nprint 2021-07-14; prior vaccination 2021-06-10; relative visit 06/30/2021\nstaff roll: Mark Holloway; Eva Burns; Nora Lee; Raj Patel\nname scatter: Ramirez, Luis; Mendes, Carlos; Stone, Emily; Lee, Marcus; Petrova, Sofia\nsymptoms log id CC-7781\n--- middle row ------------------------------------------------\nconsent line: Ramirez, Zoe {iso:2021-07-20} sig on file\n--------------------------------------------------------------\nother timestamps: 2021/07/18; 07-20-21; 2021-01-01 (policy)\n"
    ],
    "date_rule": "Choose the YYYY-MM-DD value inside the braces after 'iso:' on the consent line.",
    "expected_output": {
      "document_main_date": "2021-07-20",
      "client_information": {
        "first_name": "Zoe",
        "last_name": "Ramirez"
      }
    }
  },
  {
    "pages": [
      "XELTRONICS CORPORATION — hiring bundle\nrev 05/11/2020 meeting; start target 06/01/2020; approvals 2020-05-09\nnames across shortlist: Bell, Karen; Park, Julia; Diaz, Omar; Young, Samuel; Novak, Diana\npipeline list: Chen, Olivia; Patel, Raj; Lee, Marcus; Ali, Hassan; Brooks, Natalie\n--- middle band ----------------------------------------------\nroster slot SE-I: Ali, Hassan #on:2020-05-12# marker SE1-B\n--------------------------------------------------------------\nother dates: 05/12/20; 2020/05/12; 12-05-2020\n"
    ],
    "date_rule": "Extract the YYYY-MM-DD date between '#on:' and '#'.",
    "expected_output": {
      "document_main_date": "2020-05-12",
      "client_information": {
        "first_name": "Hassan",
        "last_name": "Ali"
      }
    }
  },
  {
    "pages": [
      "GREENFIELD UNIVERSITY — decision file\nprint 2019-03-10; committee 11/03/2019; orientation 2019-08-26; deadline 04/15/2019\nstaff: Harold King; Maya Singh; Raj Patel; Olivia Chen\nnames in cohort: Cole, Jason; Lee, Marcus; Brooks, Natalie; Mendes, Carlos; Nair, Priya\n--- central cut ----------------------------------------------\nfile tag GU-2019-4412: Petrova, Sofia <<2019-03-12>> status ADMIT\n--------------------------------------------------------------\nmirrored dates: 03-12-2019; 2019/03/12\n"
    ],
    "date_rule": "Use the date inside the double angle brackets '<<' and '>>'.",
    "expected_output": {
      "document_main_date": "2019-03-12",
      "client_information": {
        "first_name": "Sofia",
        "last_name": "Petrova"
      }
    }
  },
  {
    "pages": [
      "SKYQUEST TRAVEL — booking ledger\nprint 2018-09-04; quote 09/02/2018; depart 2018-10-05 07:25; return 2018-10-12 19:40\nagent: Irene Zhao; group coord: Hannah Park\nnames log: Cho, Daniel; Nguyen, Cathy; Cole, Jason; Petrova, Sofia; Mendes, Carlos\nPNR refs: LEE/MARCUS; CHO/DANIEL; NGUYEN/CATHY\n--- mid block -------------------------------------------------\nrecord LEE/MARCUS (iso) 2018-09-03 (iso) JFK leg confirm\n--------------------------------------------------------------\nother styles: 09-03-2018; 2018/09/03\n"
    ],
    "date_rule": "Extract the YYYY-MM-DD that appears between the tokens '(iso) ' and ' (iso)'.",
    "expected_output": {
      "document_main_date": "2018-09-03",
      "ClientInformation": {
        "first_name": "Marcus",
        "last_name": "Lee"
      }
    }
  },
  {
    "pages": [
      "RIVERSIDE ENERGY — statement dump\nperiod 11/01/2017–11/30/2017; read 2017-11-28; prior payment 2017-11-10; rebate 2017-10-05\ntouchpoints: Olga Ivanov; Sean Murphy; Emily Stone; Oliver Grant; Priya Nair\naccount roster sample: Murphy, Sean; Grant, Oliver; Nair, Priya; Brooks, Natalie; Lee, Marcus\n--- center line ----------------------------------------------\nacct row Orchard Ave: Brooks, Natalie [bill.iso=2017-12-01] due 12/21/2017\n--------------------------------------------------------------\nfooter echoes: 2017/12/01; 01-12-2017; Dec 1, 2017\n"
    ],
    "date_rule": "Use the date that appears after 'bill.iso=' inside the square brackets.",
    "expected_output": {
      "document_main_date": "2017-12-01",
      "client_information": {
        "first_name": "Natalie",
        "last_name": "Brooks"
      }
    }
  }
]