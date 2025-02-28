import os
import requests
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import json

# Global constants
ARTICLE_DIR = Path('./article')

class GetHot:
    def __init__(self, url: str):
        self.url = url
        self.list: List[Dict[str, Any]] = []
        
    def format_list(self, origin: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format the raw data into a clean list of questions"""
        return [
            {
                "title": item['target']['title'],
                "excerpt": item['target']['excerpt'],
                "url": f"https://www.zhihu.com/question/{item['target']['id']}"
            }
            for item in origin
        ]

    def get_list(self) -> List[Dict[str, str]]:
        """Fetch hot questions from Zhihu API"""
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            self.list = response.json()['data']
            return self.format_list(self.list)
        except requests.RequestException as e:
            print(f"Failed to fetch data: {e}")
            return []

    def read_old_list(self, file_path: Path) -> List[Dict[str, str]]:
        """Read existing data from file"""
        try:
            return json.loads(file_path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file: {e}")
            return []

    def write_list(self) -> None:
        """Write hot questions to file"""
        ARTICLE_DIR.mkdir(exist_ok=True)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        file_path = ARTICLE_DIR / f"{current_date}.json"
        
        current_list = self.get_list()
        if not current_list:
            print("No data to write")
            return

        old_list = self.read_old_list(file_path)
        
        # Merge lists and remove duplicates while preserving order
        merged_list = old_list + current_list
        seen = set()
        result = []
        for item in merged_list:
            item_tuple = tuple(sorted(item.items()))
            if item_tuple not in seen:
                seen.add(item_tuple)
                result.append(item)

        try:
            file_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=4),
                encoding='utf-8'
            )
            print(f"Successfully wrote to {file_path}")
        except IOError as e:
            print(f"Error writing to file: {e}")
        
        # Generate daily MD file
        self.create_readme(result)

    def create_readme(self, questions: List[Dict[str, str]]) -> None:
        """Generate daily MD file with current hot questions in article directory"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        readme_content = f"""# {current_date} 知乎热门话题

共 {len(questions)} 条

{self.create_readme_list(questions)}
"""
        try:
            ARTICLE_DIR.mkdir(exist_ok=True)
            file_path = ARTICLE_DIR / f"{current_date}.md"
            file_path.write_text(readme_content, encoding='utf-8')
            print(f"Successfully created {file_path}")
        except IOError as e:
            print(f"Error writing daily MD file: {e}")

    def create_readme_list(self, questions: List[Dict[str, str]]) -> str:
        """Format questions list for README"""
        question_list = "\n".join(
            f"1. [{q['title']}]({q['url']})" 
            for q in questions
        )
        return f"<!-- BEGIN -->\n<!-- 最后更新时间 {datetime.now()} -->\n{question_list}\n<!-- END -->"


if __name__ == "__main__":
    url = 'https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit=100'
    hot = GetHot(url)
    hot.write_list()