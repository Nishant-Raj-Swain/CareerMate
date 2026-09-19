import httpx
from bs4 import BeautifulSoup
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

CATEGORY_URLS = {
    # Technical / Programming
    "python": "https://www.indiabix.com/python-programming/classes/022001",
    "c": "https://www.indiabix.com/c-programming/questions-and-answers/001001",
    "java": "https://www.indiabix.com/java-programming/questions-and-answers/010001",
    "networking": "https://www.indiabix.com/networking/questions-and-answers/001001",
    "sql": "https://www.indiabix.com/database/questions-and-answers/001001",

    # Quantitative Aptitude
    "aptitude": "https://www.indiabix.com/aptitude/average/003001",
    "profit-loss": "https://www.indiabix.com/aptitude/profit-and-loss/005001",
    "average": "https://www.indiabix.com/aptitude/average/003001",
    "time-work": "https://www.indiabix.com/aptitude/time-and-work/013001",

    # Logical Reasoning
    "reasoning": "https://www.indiabix.com/logical-reasoning/cause-and-effect/032001",
    "logical": "https://www.indiabix.com/logical-reasoning/cause-and-effect/032001",
    "cause-effect": "https://www.indiabix.com/logical-reasoning/cause-and-effect/032001",
    "statement-conclusion": "https://www.indiabix.com/logical-reasoning/statement-and-conclusion/036001"
}

async def fetch_indiabix_questions(topic: str = "python", total_needed: int = 20) -> list[dict]:
    topic_clean = topic.strip().lower()
    start_url = CATEGORY_URLS.get(topic_clean, CATEGORY_URLS["python"])

    extracted_data = []
    
    # Construct paginated URLs (022001 -> 022002, 022003...)
    base_match = re.search(r'^(.*?/(\d+))(\d{3})$', start_url)
    if not base_match:
        urls_to_scrape = [start_url]
    else:
        prefix_url = base_match.group(1)
        start_num = int(base_match.group(3))
        urls_to_scrape = [f"{prefix_url}{start_num + i:03d}" for i in range(5)]

    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        for url in urls_to_scrape:
            if len(extracted_data) >= total_needed:
                break

            try:
                response = await client.get(url, headers=HEADERS)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                questions = soup.find_all('div', class_='bix-div-container')

                for q in questions:
                    if len(extracted_data) >= total_needed:
                        break

                    # Question text
                    q_text_div = q.find('div', class_='bix-td-qtxt')
                    q_text = q_text_div.get_text(separator=" ", strip=True) if q_text_div else ""

                    # Options
                    options = [opt.get_text(strip=True) for opt in q.find_all('div', class_='bix-td-option-val')]
                    if len(options) < 2:
                        continue

                    # Answer Letter Extraction
                    correct_letter = None
                    ans_input = q.find('input', class_='jq-has-open-qno') or q.find('input', class_='jq-hd-val-bg')
                    if ans_input and ans_input.get('value'):
                        val = ans_input.get('value').strip().upper()
                        if val in ['A', 'B', 'C', 'D']:
                            correct_letter = val

                    if not correct_letter:
                        for inp in q.find_all('input', type='hidden'):
                            val = inp.get('value', '').strip().upper()
                            if val in ['A', 'B', 'C', 'D']:
                                correct_letter = val
                                break

                    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
                    correct_index = mapping.get(correct_letter, 0)

                    extracted_data.append({
                        "question": q_text,
                        "options": options[:4],
                        "correct": correct_index
                    })

            except Exception as e:
                print(f"[Error fetching {url}]: {e}")

    return extracted_data