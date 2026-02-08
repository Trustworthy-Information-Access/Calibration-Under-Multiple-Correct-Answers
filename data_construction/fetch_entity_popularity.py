import requests
import json
import argparse
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 

# --- API Endpoints ---
WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_PAGEVIEWS_API_ENDPOINT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{article}/monthly/{start}/{end}"

# --- Session for connection pooling ---
SESSION = requests.Session()
HEADERS = {
    'User-Agent': 'MyPopularityScraper/1.2 (https://example.com/info; my-email@example.com)'
}
SESSION.headers.update(HEADERS)

def retry_request(func, max_retries=3, sleep_time=2, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"   - Attempt {attempt+1} failed, retrying... ({e})")
                time.sleep(sleep_time)
            else:
                print(f"   - Failed after multiple retries: {e}", file=sys.stderr)
    return None

def get_page_views_12m(page_title: str) -> int:
    """Get pageviews for the past 12 months"""
    end_date = datetime.now(timezone.utc).replace(day=1) - relativedelta(days=1)
    start_date = end_date - relativedelta(months=11)
    start_str = start_date.strftime('%Y%m01')
    end_str = end_date.strftime('%Y%m%d')
    url = WIKIPEDIA_PAGEVIEWS_API_ENDPOINT.format(article=page_title, start=start_str, end=end_str)
    
    def do_request():
        response = SESSION.get(url)
        if response.status_code == 404: return 0
        response.raise_for_status()
        data = response.json()
        return sum(item.get("views", 0) for item in data.get("items", []))
    
    try:
        result = retry_request(do_request, max_retries=3, sleep_time=2)
        if result is None:
            return 0
        return result
    except Exception:
        return 0

def get_instances_with_sparql(class_qid: str, limit: int = 5000) -> list:
    print(f"1. Querying instances for class '{class_qid}' (batch size {limit})...")
    all_qids = []
    offset = 0
    # For simplicity, only do one query here. Remove comment for while True loop if pagination needed.
    query = f"""
    SELECT ?item WHERE {{
        ?item wdt:P31 wd:{class_qid}.
    }}
    LIMIT {limit}
    OFFSET {offset}
    """
    params = {'query': query, 'format': 'json'}
    def do_request():
        response = SESSION.get(WIKIDATA_SPARQL_ENDPOINT, params=params, headers={'Accept': 'application/sparql-results+json'})
        response.raise_for_status()
        data = response.json()
        return data
    
    data = retry_request(do_request, max_retries=3, sleep_time=2)
    if data is not None:
        batch_qids = [item['item']['value'].split('/')[-1] for item in data['results']['bindings']]
        all_qids.extend(batch_qids)
        print(f"   - SPARQL query succeeded, got {len(all_qids)} instances currently.")
    else:
        print(f"   - SPARQL request failed after multiple retries", file=sys.stderr)
    
    return all_qids

def get_details_for_qids(qids: list[str]) -> dict:
    print(f"2. Batch retrieving details for {len(qids)} instances...")
    details = {}
    
    for i in tqdm(range(0, len(qids), 50), desc="Fetching details", unit="batch"):
        chunk_qids = qids[i:i+50]
        params = {
            "action": "wbgetentities", "format": "json", "ids": "|".join(chunk_qids),
            "props": "labels|sitelinks", "languages": "en", "sitefilter": "enwiki"
        }
        def do_request():
            response = SESSION.get(WIKIDATA_API_ENDPOINT, params=params)
            response.raise_for_status()
            return response.json()
        
        data = retry_request(do_request, max_retries=3, sleep_time=2)
        if data is not None:
            entities = data.get("entities", {})
            for qid, entity_data in entities.items():
                label = entity_data.get("labels", {}).get("en", {}).get("value")
                sitelink = entity_data.get("sitelinks", {}).get("enwiki", {})
                title = sitelink.get("title")
                if label and title:
                    formatted_title = title.replace(" ", "_")
                    details[qid] = {"label": label, "title": formatted_title}
        else:
            print(f"   - API request failed after multiple retries", file=sys.stderr)
            time.sleep(1)
            
    valid_details_count = len(details)
    print(f"   - Successfully retrieved details for {valid_details_count} instances (some without English Wikipedia page may be filtered).")
    return details

def get_popularity_data_for_class(class_qid: str, outfile: str, limit: int, entity_name: str):
    """
    General function to fetch popularity data for a class
    """
    instance_qids = get_instances_with_sparql(class_qid, limit=limit)
    if not instance_qids:
        print("No instances found, aborting.")
        return

    details = get_details_for_qids(instance_qids)

    print(f"3. Fetching 12-month pageviews for each instance (total: {len(details)})...")
    final_list = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(get_page_views_12m, detail["title"]): (qid, detail) for qid, detail in details.items()}
        for future in as_completed(futures):
            qid, detail_data = futures[future]
            try:
                views = future.result()
                if views > 0:
                    final_list.append({
                        "qid": qid,
                        "label": detail_data["label"],
                        "title": detail_data["title"],
                        "views_12m": views
                    })
            except Exception as e:
                print(f"Error on {detail_data['title']}: {e}")
                
    final_list = [item for item in final_list if item["views_12m"] > 0]
    print("\n4. All data processed, sorting and saving...")
    final_list.sort(key=lambda x: x['views_12m'], reverse=True)
    
    try:
        as_of = datetime.now().strftime('%Y-%m')
        output_data = {
            "as_of": as_of,
            entity_name: final_list
        }
        with open(outfile, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Successfully saved result to file: {outfile}")
    except IOError as e:
        print(f"❌ Failed to save file: {e}", file=sys.stderr)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetch Wikidata entity popularity data')
    parser.add_argument('--limit', type=int, default=5000, 
                       help='Query limit per class (default: 5000)')
    parser.add_argument('--output_dir', type=str, default='./', 
                       help='Output directory path (default: current directory)')
    parser.add_argument('--entity', type=str, default=None, 
                       help='Specify entity types to process, comma separated (default: all supported)')
    parser.add_argument('--list_entities', action='store_true',
                       help='List all supported entity types')
    return parser.parse_args()

# --- Main program entrypoint ---
if __name__ == "__main__":
    args = parse_arguments()
    
    # Note: The key and outfile for Country have been changed to "region"
    ENTITY_CONFIG = {
        "region":   {"qid": "Q6256",    "limit": args.limit, "outfile": "region_popularity.json"},
        "award":    {"qid": "Q618779",  "limit": args.limit, "outfile": "award_popularity.json"},
        "office":   {"qid": "Q4164871", "limit": args.limit, "outfile": "office_popularity.json"},
        "river":    {"qid": "Q4022",    "limit": args.limit, "outfile": "river_popularity.json"},
        "language": {"qid": "Q34770",   "limit": args.limit, "outfile": "language_popularity.json"},
    }
    
    if args.list_entities:
        print("Supported entity types:")
        for entity_type, config in ENTITY_CONFIG.items():
            print(f"  {entity_type}: {config['qid']} - {config['outfile']}")
        sys.exit(0)
    
    if args.entity:
        specified_entities = [e.strip() for e in args.entity.split(',')]
        invalid_entities = [e for e in specified_entities if e not in ENTITY_CONFIG]
        if invalid_entities:
            print(f"❌ Unsupported entity type(s): {', '.join(invalid_entities)}")
            print(f"Supported entity types: {', '.join(ENTITY_CONFIG.keys())}")
            sys.exit(1)
        ENTITY_CONFIG = {k: v for k, v in ENTITY_CONFIG.items() if k in specified_entities}
        print(f"📋 Will process the following entity types: {', '.join(ENTITY_CONFIG.keys())}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📁 Created output directory: {args.output_dir}")
    
    print(f"⚙️  Configuration:")
    print(f"   - Query limit: {args.limit}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Entity types: {', '.join(ENTITY_CONFIG.keys())}")
    
    for entity_type, config in ENTITY_CONFIG.items():
        if config.get("qid") and config.get("outfile"):
            print(f"\n{'='*20} Start processing type: {entity_type.upper()} {'='*20}")
            output_file = os.path.join(args.output_dir, config["outfile"])
            
            get_popularity_data_for_class(
                class_qid=config["qid"],
                outfile=output_file,
                limit=config.get("limit", args.limit),
                entity_name=entity_type
            )