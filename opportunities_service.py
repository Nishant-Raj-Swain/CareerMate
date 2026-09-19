import httpx
import logging

logging.basicConfig(level=logging.INFO)

class OpportunitiesService:
    def __init__(self):
        self.api_url = "https://unstop.com/api/public/opportunity/search-result"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://unstop.com/",
        }

    async def fetch_opportunities(self, opportunity_type: str = "competitions", limit: int = 5) -> str:
        """
        Fetches open competitions or internships from Unstop for Engineering/IT (domain=2).
        opportunity_type: 'competitions' or 'internships'
        """
        params = {
            "opportunity": opportunity_type,
            "oppstatus": "open",
            "usertype": "students",
            "domain": 2,  # Engineering / IT Domain
            "per_page": limit,
            "page": 1,
        }

        async with httpx.AsyncClient(timeout=15.0, headers=self.headers) as client:
            try:
                response = await client.get(self.api_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    opportunities = data.get("data", {}).get("data", [])
                    
                    if not opportunities:
                        return f"No active {opportunity_type} found on Unstop right now."

                    title_header = "🏆 *Latest Competitions on Unstop*" if opportunity_type == "competitions" else "💼 *Latest Internships on Unstop*"
                    message_text = f"{title_header}\n\n"

                    for idx, opp in enumerate(opportunities[:limit], start=1):
                        title = opp.get("title", "N/A")
                        org = opp.get("organisation", {}).get("name", "N/A")
                        seo_url = opp.get("seo_url", "").strip("/")
                        
                        if seo_url.startswith("http"):
                            link = seo_url
                        elif seo_url:
                            link = f"https://unstop.com/{opportunity_type}/{seo_url}"
                        else:
                            link = "https://unstop.com"

                        message_text += f"*{idx}. {title}*\n"
                        message_text += f"🏢 *Org:* {org}\n"
                        message_text += f"🔗 *Link:* {link}\n"
                        message_text += "-----------------------------------\n"

                    return message_text
                else:
                    return f"Failed to fetch {opportunity_type}. HTTP Status: {response.status_code}"

            except Exception as e:
                logging.error(f"Error fetching Unstop data: {e}")
                return f"An error occurred while fetching {opportunity_type}."

# Instance to import directly
opportunities_service = OpportunitiesService()