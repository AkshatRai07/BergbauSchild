import re
import json
import pandas as pd
import pdfplumber
from typing import List, Dict, Any

def extract_accident_records(pdf_path: str) -> List[Dict[str, Any]]:

    accidents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    code_pattern = r'Code\s*:\s*(\d+)\s+(.+?)(?=\n)'
    code_matches = list(re.finditer(code_pattern, full_text))

    code_map = {}
    for match in code_matches:
        code_map[match.start()] = {
            'code': match.group(1),
            'category': match.group(2).strip()
        }

    pattern = r'(\d+)\.\s+Date\s*-\s*(\d{2}/\d{2}/\d{2})\s+Mine\s*-\s*(.+?)\n\s*Time\s*-\s*([\d\.]+)\s+Owner\s*-\s*(.+?)\n\s*Dist\.\s*-\s*(.+?),\s*State\s*-\s*(.+?)(?=\n)'
    
    matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
    
    print(f"Found {len(matches)} accident records")
    
    for i, match in enumerate(matches):
        try:
            accident_id = int(match.group(1))
            start_pos = match.start()

            accident_code = ""
            accident_category = ""
            for code_pos in sorted(code_map.keys(), reverse=True):
                if code_pos < start_pos:
                    accident_code = code_map[code_pos]['code']
                    accident_category = code_map[code_pos]['category']
                    break

            if i < len(matches) - 1:
                end_pos = matches[i+1].start()
            else:
                end_pos = len(full_text)
            
            accident_text = full_text[start_pos:end_pos]

            victims = []
            victim_pattern = r'(\d+)\.\s*([^,\n]+),\s*([^,\n]+),\s*(Male|Female),\s*(\d+)\s*Years'
            for vm in re.finditer(victim_pattern, accident_text, re.IGNORECASE):
                victims.append({
                    'name': vm.group(2).strip(),
                    'occupation': vm.group(3).strip(),
                    'gender': vm.group(4).strip(),
                    'age': int(vm.group(5))
                })

            desc_pattern = r'Years\s*\n\s*(.+?)(?=\n\s*Had|$)'
            desc_match = re.search(desc_pattern, accident_text, re.DOTALL | re.IGNORECASE)
            incident_description = desc_match.group(1).strip() if desc_match else ""

            incident_description = re.sub(r'\s+', ' ', incident_description)

            root_cause_pattern = r'Had\s+(.+?)(?=this accident could have been averted|this accident would have been averted|$)'
            root_match = re.search(root_cause_pattern, accident_text, re.DOTALL | re.IGNORECASE)
            root_cause = root_match.group(1).strip() if root_match else ""

            root_cause = re.sub(r'\s+', ' ', root_cause)

            regulations = []
            reg_pattern = r'Regulation\s+([\d\(\)]+[A-Za-z]*)'
            for rm in re.finditer(reg_pattern, accident_text):
                reg_text = rm.group(0)
                if reg_text not in regulations:
                    regulations.append(reg_text)

            mine_type = "unknown"
            desc_lower = incident_description.lower()
            if "underground" in desc_lower:
                mine_type = "underground"
            elif "opencast" in desc_lower or "open cast" in desc_lower:
                mine_type = "opencast"
            elif "quarry" in desc_lower:
                mine_type = "opencast"

            height_pattern = r'(\d+(?:\.\d+)?)\s*m\s*(?:high|height)'
            height_matches = re.findall(height_pattern, desc_lower)
            fall_height = height_matches[0] + "m" if height_matches else ""

            accident_record = {
                'accident_id': accident_id,
                'date': match.group(2),
                'time': match.group(4).strip(),
                'mine_name': match.group(3).strip(),
                'owner': match.group(5).strip(),
                'district': match.group(6).strip(),
                'state': match.group(7).strip(),
                'deaths_count': len(victims),
                'accident_code': accident_code,
                'accident_category': accident_category,
                'mine_type': mine_type,
                'incident_description': incident_description,
                'root_cause': root_cause,
                'regulations_violated': "; ".join(regulations),
                'fall_height': fall_height,
            }

            if victims:
                accident_record['victim_name'] = "; ".join([v['name'] for v in victims])
                accident_record['victim_occupation'] = "; ".join([v['occupation'] for v in victims])
                accident_record['victim_gender'] = "; ".join([v['gender'] for v in victims])
                accident_record['victim_age'] = "; ".join([str(v['age']) for v in victims])
            else:
                accident_record['victim_name'] = ""
                accident_record['victim_occupation'] = ""
                accident_record['victim_gender'] = ""
                accident_record['victim_age'] = ""
            
            accidents.append(accident_record)
            
        except Exception as e:
            print(f"Error processing accident {accident_id if 'accident_id' in locals() else 'unknown'}: {str(e)}")
            continue
    
    return accidents


def extract_accident_codes(pdf_path: str) -> Dict[str, str]:
    """Extract accident code mappings from the PDF"""
    codes = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text and "Code" in text and "Cause of Accident" in text:
                full_text += text + "\n"

        pattern = r'(\d{4})\s+(.+?)(?=\n\s*\d{4}|\Z)'
        matches = re.finditer(pattern, full_text, re.DOTALL)
        
        for match in matches:
            code = match.group(1)
            description = match.group(2).strip()
            description = re.sub(r'\s+', ' ', description)
            description = re.sub(r'\n', ' ', description)
            codes[code] = description
    
    return codes


def process_mining_accidents_pdf(pdf_path: str, output_csv: str = "mining_accidents.csv"):
    """Main processing function"""
    print(f"Processing PDF: {pdf_path}")

    print("\nExtracting accident records...")
    accidents = extract_accident_records(pdf_path)
    print(f"Extracted {len(accidents)} accident records")
    
    if len(accidents) == 0:
        print("\nWARNING: No accidents were extracted. Check the PDF format.")
        return None, None

    print("\nExtracting accident codes...")
    codes = extract_accident_codes(pdf_path)
    print(f"Extracted {len(codes)} accident codes")

    df = pd.DataFrame(accidents)

    print("\n=== Sample Data ===")
    print(df[['accident_id', 'date', 'mine_name', 'accident_category', 'deaths_count']].head(10))

    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nSaved {len(df)} records to {output_csv}")

    codes_json = output_csv.replace('.csv', '_codes.json')
    with open(codes_json, 'w', encoding='utf-8') as f:
        json.dump(codes, f, indent=2, ensure_ascii=False)
    print(f"Saved accident codes to {codes_json}")

    print("\n=== Statistics ===")
    print(f"Total accidents: {len(df)}")
    print(f"Total deaths: {df['deaths_count'].sum()}")
    print(f"\nAccidents by category:")
    print(df['accident_category'].value_counts())
    print(f"\nAccidents by state:")
    print(df['state'].value_counts())
    
    return df, codes


if __name__ == "__main__":
    df, codes = process_mining_accidents_pdf("VOLUME_II_NON_COAL_2015.pdf")
    
    if df is not None:
        print("\n=== Processing Complete ===")
        print(f"Output files created successfully!")