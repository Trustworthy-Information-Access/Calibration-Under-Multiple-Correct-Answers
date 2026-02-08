import json
import time
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Obtain structured facts and sub-entity mappings for entities from Wikidata')
    parser.add_argument('--entity_popularity_dir', type=str, 
                       default='/mnt/public/gpfs-jd/data/lh/wyh/Multi_Answer_Confidence/generate_data/new_code/entity_popularity',
                       help='Path to entity popularity data directory (default: entity_popularity directory)')
    parser.add_argument('--output_dir', type=str, 
                       default='/mnt/public/gpfs-jd/data/lh/wyh/Multi_Answer_Confidence/generate_data/new_code/entity_structure&map',
                       help='Path to output directory (default: entity_structure&map directory)')
    parser.add_argument('--entity', type=str, default="all", 
                       help='Specify which entity types to process, separate multiple types with commas (default: all)')
    parser.add_argument('--limit', type=int, default=2000, 
                       help='SPARQL query LIMIT (default: 2000)')
    parser.add_argument('--list_entities', action='store_true',
                       help='List all supported entity types')
    parser.add_argument('--retry', type=int, default=3, help='Max SPARQL query retries (default 3)')
    parser.add_argument('--top_k', type=int, default=5000, help='Max top_k entities (default 5k)')
    parser.add_argument('--timeout', type=int, default=30, help='SPARQL single query timeout in seconds (default 30)')
    parser.add_argument('--outer_workers', type=int, default=32, help='Outer thread concurrency (default 32)')
    parser.add_argument('--inner_workers', type=int, default=32, help='Inner thread concurrency (default 32)')
    parser.add_argument('--view_num', type=int, default=1, help='Minimum view_num threshold for filtering entities (default 1)')
    return parser.parse_args()


WDQS = "https://query.wikidata.org/sparql"
TOP_K = 2000

# Default concurrency increased
OUTER_WORKERS = 64
INNER_WORKERS = 64

def get_sparql(timeout=30):
    sparql = SPARQLWrapper(WDQS, agent="MultiAnswerQA/qa-dataset-builder 1.0")
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    return sparql

def run_query(q, retries=10, timeout=30):
    """Multiple retries to ensure completeness, auto switch User-Agent and adjust timeout"""
    for i in range(retries):
        try:
            sparql = get_sparql(timeout)
            sparql.setQuery(q)
            return sparql.query().convert()
        except Exception as e:
            # Shorter sleep time, speed up retry on failure
            time.sleep(0.5 + 0.5 * i)
    return None

def get_count_for_query(q, retries=3, timeout=30):
    """Return total count (COUNT) for a given SPARQL query"""
    import re
    q_no_order = re.sub(r'ORDER BY[^\n]*', '', q, flags=re.IGNORECASE)
    q_no_limit = re.sub(r'LIMIT\s+\d+', '', q_no_order, flags=re.IGNORECASE)
    q_count = re.sub(r'SELECT\s+[^{]*WHERE', 'SELECT (COUNT(*) AS ?count) WHERE', q_no_limit, flags=re.IGNORECASE)
    result = run_query(q_count, retries=retries, timeout=timeout)
    if not result:
        return None
    try:
        bindings = result.get("results", {}).get("bindings", [])
        if bindings and "count" in bindings[0]:
            return int(bindings[0]["count"]["value"])
    except Exception:
        pass
    return None

# ==================== fetch_and_validate - universal querying and validation ====================
def fetch_and_validate(q, qid, parent_qid, sub_label, limit, years_key="years", extra_keys=None):
    """
    Universal fetching and validation function for all querying operations
    """
    total_count = get_count_for_query(q)
    if total_count is not None and total_count > limit:
        result = {years_key: {}, "skipped": True, "total_count": total_count}
        result.update({"qid": qid, "parent_qid": parent_qid})
        if extra_keys:
            result.update(extra_keys)
        return sub_label, result

    result = run_query(q)
    # Use retry mechanism to re-fetch if needed
    max_retry = 10
    for attempt in range(max_retry):
        result_count = len(result["results"]["bindings"]) if result and "results" in result and "bindings" in result["results"] else 0
        if total_count is not None and result_count != total_count:
            result = run_query(q)
        else:
            break

    if not result:
        result = {years_key: {}}
        result.update({"qid": qid, "parent_qid": parent_qid})
        if extra_keys:
            result.update(extra_keys)
        return sub_label, result

    return result

def run_paged_query(qid, query_template, limit, years_key):
    """
    Universal paged query tool, automatically paginates to fetch all results.
    (Originally in Mountain strategy, but used by River too, thus moved to common section)
    """
    offset = 0
    all_bindings = []
    while True:
        q = query_template.format(qid=qid, limit=limit, offset=offset)
        fetch_result = fetch_and_validate(q, qid, qid, years_key, limit, years_key=years_key)

        if isinstance(fetch_result, tuple):
            _, result = fetch_result
            if result.get("skipped"):
                break
            return []  # Return empty list on error

        result = fetch_result
        if not result or "results" not in result or "bindings" not in result["results"]:
            break

        bindings = result["results"]["bindings"]
        if not bindings:
            break

        all_bindings.extend(bindings)

        # If results < limit, fetching is complete
        if len(bindings) < limit:
            break

        offset += limit

    return all_bindings


# ==================== AWARD strategy ====================
def get_sub_awards(qid, limit=100, timeout=30):
    data_query = f"""
    SELECT DISTINCT ?sub ?subLabel WHERE {{
      {{ ?sub wdt:P361 wd:{qid}. }} UNION {{ wd:{qid} wdt:P527 ?sub. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """
    # Use fetch_and_validate
    fetch_result = fetch_and_validate(data_query, qid, qid, "sub_awards", limit, years_key="sub_awards")
    if isinstance(fetch_result, tuple):
        # Incomplete results or failed
        _, result = fetch_result
        if result.get("skipped"):
            return []
        # Failed
        return []
    result = fetch_result
    bindings = result.get("results", {}).get("bindings", []) if result else []
    if not bindings:
        return []
    return [
        {"qid": b["sub"]["value"].split("/")[-1],
         "label": b.get("subLabel", {}).get("value", b["sub"]["value"].split("/")[-1])}
        for b in bindings
    ]

def fetch_award_recipients(qid, parent_qid, sub_label, limit):
    q = f"""
    SELECT DISTINCT ?entity ?entityLabel ?date ?country ?countryLabel WHERE {{
      ?entity p:P166 ?stmt .
      ?stmt ps:P166 wd:{qid} .

      # Only human beings
      ?entity wdt:P31/wdt:P279* wd:Q5 .

      # Exclude deprecated ranks
      ?stmt wikibase:rank ?rank .
      FILTER(?rank != wikibase:DeprecatedRank)

      # Award date (if exists)
      OPTIONAL {{ ?stmt pq:P585 ?date. }}

      # Country (if exists)
      OPTIONAL {{ ?entity wdt:P27 ?country. }}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """
    fetch_result = fetch_and_validate(q, qid, parent_qid, sub_label, limit, years_key="years")
    if isinstance(fetch_result, tuple):
        return fetch_result

    result = fetch_result
    year_map = {}
    for b in result["results"]["bindings"]:
        try:
            ent = b["entity"]["value"].split("/")[-1]
            label = b.get("entityLabel", {}).get("value", ent)

            country_uri = b.get("country", {}).get("value", "")
            country_qid = country_uri.split("/")[-1] if country_uri else ""
            country_label = b.get("countryLabel", {}).get("value", country_qid)

            if "date" in b:
                year_str = b["date"]["value"][:4]
                year = int(year_str)
                key = year_str if 1800 <= year <= datetime.now().year else "invalid_year"
            else:
                key = "unknown"

            info = [label, ent, "human", country_label, country_qid]
            year_map.setdefault(key, []).append(info)
        except Exception:
            continue

    # Sort so that "unknown" is at the end
    def _key(k):
        return (k == "unknown", 9999 if not k.isdigit() else int(k))
    ordered = dict(sorted(year_map.items(), key=lambda kv: _key(kv[0])))

    return sub_label, {"qid": qid, "parent_qid": parent_qid, "years": ordered}

def process_award(award, limit):
    label, qid = award["label"], award["qid"]
    # Keep view_12m
    view_12m = award.get("views_12m", None)
    sub_awards = get_sub_awards(qid, limit)
    sub_awards.insert(0, {"qid": qid, "label": label})
    mapping = {qid: list({s["qid"] for s in sub_awards})}

    facts = {}
    with ThreadPoolExecutor(max_workers=INNER_WORKERS) as inner_pool:
        futures = [
            inner_pool.submit(fetch_award_recipients, s["qid"], qid, s["label"], limit)
            for s in sub_awards
        ]
        for f in as_completed(futures):
            sub_label, fact = f.result()
            # Keep view_12m
            if view_12m is not None:
                fact["views_12m"] = view_12m
            facts[sub_label] = fact

    return mapping, facts

# ==================== COUNTRY strategy ====================
def fetch_admin_divisions_as_of(qid, label, limit=200, view_12m=None, as_of_date="2023-12-31"):
    """
    Only return ADM1 ("first-level administrative divisions") that are still in effect as of as_of_date.
    If there are no first-level divisions (or query is empty/failed), return {} (remove the country).
    """
    cutoff_end = f"{as_of_date}T23:59:59Z"

    q = f"""
    SELECT DISTINCT ?sub ?subLabel ?typeLabel ?area ?population ?start ?end WHERE {{
      wd:{qid} p:P150 ?stmt .
      ?stmt ps:P150 ?sub .
      ?stmt wikibase:rank ?rank .
      VALUES ?rank {{ wikibase:PreferredRank wikibase:NormalRank }}

      OPTIONAL {{ ?stmt pq:P580 ?start. }}
      OPTIONAL {{ ?stmt pq:P582 ?end. }}
      BIND("{cutoff_end}"^^xsd:dateTime AS ?cutoff)

      # Validity: start ≤ cutoff and (no end or end ≥ cutoff)
      FILTER( (!BOUND(?start) || ?start <= ?cutoff) && (!BOUND(?end) || ?end >= ?cutoff) )

      # Restrict to first-level administrative subdivision and its subclasses
      ?sub wdt:P31/wdt:P279* wd:Q10864048 .

      # Optional: If sub-entity claims P17, it must be the same country; if no P17, allow
      OPTIONAL {{ ?sub wdt:P17 ?country. }}
      FILTER(!BOUND(?country) || ?country = wd:{qid})

      OPTIONAL {{ ?sub wdt:P31 ?type. }}
      OPTIONAL {{ ?sub wdt:P2046 ?area. }}
      OPTIONAL {{ ?sub wdt:P1082 ?population. }}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,zh,zh-hans,zh-hant". }}
    }}
    ORDER BY LCASE(STR(?subLabel))
    LIMIT {limit}
    """

    fetch_result = fetch_and_validate(q, qid, qid, label, limit, years_key="divisions")

    # Query fails or structure abnormal => remove this country (return empty dict)
    if (not fetch_result or
        "results" not in fetch_result or
        "bindings" not in fetch_result["results"]):
        return {}

    divisions = []
    for b in fetch_result["results"]["bindings"]:
        try:
            sub_qid = b["sub"]["value"].split("/")[-1]
            sub_name = b.get("subLabel", {}).get("value", sub_qid)
            sub_type = b.get("typeLabel", {}).get("value", "")
            area = b.get("area", {}).get("value", None)
            population = b.get("population", {}).get("value", None)

            divisions.append({
                "name": sub_name,
                "qid": sub_qid,
                "type": sub_type,
                "area": float(area) if area else None,
                "population": int(float(population)) if population else None
            })
        except Exception:
            continue

    # If no ADM1, remove the country (return empty dict)
    if not divisions:
        return {}

    # Only return when ADM1 exists
    ret = {label: {"qid": qid, "parent_qid": qid, "divisions": divisions}}
    if view_12m is not None:
        ret[label]["views_12m"] = view_12m
    return ret

def process_country(country, limit):
    label, qid = country["label"], country["qid"]
    view_12m = country.get("views_12m", None)
    facts = fetch_admin_divisions_as_of(qid, label, limit, view_12m=view_12m)
    return {}, facts

# ==================== LANGUAGE strategy ====================
def get_sub_languages(qid, limit):
    q = f"""
    SELECT DISTINCT ?sub ?subLabel WHERE {{
      {{ ?sub wdt:P279 wd:{qid}. }} UNION
      {{ ?sub wdt:P361 wd:{qid}. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}"""
    fetch_result = fetch_and_validate(q, qid, qid, "sub_languages", limit, years_key="sub_languages")
    if isinstance(fetch_result, tuple):
        _, result = fetch_result
        if result.get("skipped"):
            return []
        return []
    result = fetch_result
    if not result or "results" not in result or "bindings" not in result["results"]:
        return []
    return [
        {"qid": b["sub"]["value"].split("/")[-1], "label": b.get("subLabel", {}).get("value", b["sub"]["value"].split("/")[-1])}
        for b in result["results"]["bindings"]
    ]


def fetch_language_countries(
    qid,
    parent_qid,
    sub_label,
    limit=200,
    view_12m=None,
    require_iso_code=True,        
):
    """
    Only keep "current sovereign states".
    Language requirement: P2936 (language used) or P37 (official language).
    Filter: remove historical/former/ended states; can optionally require P297 (ISO 3166-1).
    """

    iso_filter = "FILTER EXISTS { ?entity wdt:P297 ?_iso }" if require_iso_code else ""

    q = f"""
    SELECT DISTINCT ?entity ?entityLabel WHERE {{
      # Language filter
      {{
        ?entity wdt:P2936 wd:{qid}.
      }} UNION {{
        ?entity wdt:P37 wd:{qid}.
      }}

      # Type restriction: sovereign state (and subclasses), not administrative region
      FILTER EXISTS {{ ?entity wdt:P31/wdt:P279* wd:Q3624078 }}   # sovereign state

      # Exclude historical/ended entities
      FILTER NOT EXISTS {{ ?entity wdt:P576 ?_dissolved }}        # dissolved/abolished date
      FILTER NOT EXISTS {{ ?entity wdt:P582 ?_ended }}            # end time
      FILTER NOT EXISTS {{ ?entity wdt:P31/wdt:P279* wd:Q3024240 }}   # former country
      FILTER NOT EXISTS {{ ?entity wdt:P31/wdt:P279* wd:Q28171280 }}  # historical polity

      # Optional: Require ISO 3166-1 code to enhance confidence in "current country"
      {iso_filter}

      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    fetch_result = fetch_and_validate(q, qid, parent_qid, sub_label, limit, years_key="countries")
    if isinstance(fetch_result, tuple):
        sub_label_ret, fact = fetch_result
        if view_12m is not None:
            fact["views_12m"] = view_12m
        return sub_label_ret, fact

    result = fetch_result
    bindings = result.get("results", {}).get("bindings", [])

    # Only country: do not return parent (to avoid mixing in administrative regions)
    countries = []
    seen = set()
    for b in bindings:
        ent_uri = b["entity"]["value"]
        ent = ent_uri.split("/")[-1]
        label = b.get("entityLabel", {}).get("value", ent)
        if ent in seen:
            continue
        seen.add(ent)
        countries.append([label, ent, "", ""])  # kept compatible with original return format: [label, qid, parent_label, parent_qid]

    ret = {"qid": qid, "parent_qid": parent_qid, "countries": countries}
    if view_12m is not None:
        ret["views_12m"] = view_12m
    return sub_label, ret


def process_language(language, limit):
    label, qid = language["label"], language["qid"]
    view_12m = language.get("views_12m", None)
    sub_languages = get_sub_languages(qid, limit)
    sub_languages.insert(0, {"qid": qid, "label": label})
    mapping = {qid: list({s["qid"] for s in sub_languages})}
    facts = {}
    with ThreadPoolExecutor(max_workers=INNER_WORKERS) as inner_pool:
        futures = [
            inner_pool.submit(fetch_language_countries, s["qid"], qid, s["label"], limit, view_12m=view_12m)
            for s in sub_languages
        ]
        for f in as_completed(futures):
            sub_label, fact = f.result()
            facts[sub_label] = fact
    return mapping, facts

# ==================== OFFICE strategy ====================
def get_sub_offices(qid, limit):
    q = f"""
    SELECT DISTINCT ?sub ?subLabel WHERE {{
      {{ ?sub wdt:P279 wd:{qid}. }} UNION
      {{ ?sub wdt:P1365 wd:{qid}. }} UNION
      {{ ?sub wdt:P1366 wd:{qid}. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}"""
    fetch_result = fetch_and_validate(q, qid, qid, "sub_offices", limit, years_key="sub_offices")
    if isinstance(fetch_result, tuple):
        _, result = fetch_result
        if result.get("skipped"):
            return []
        return []
    result = fetch_result
    if not result or "results" not in result or "bindings" not in result["results"]:
        return []
    sub_offices = [
        {
            "qid": b["sub"]["value"].split("/")[-1], 
            "label": b.get("subLabel", {}).get("value", b["sub"]["value"].split("/")[-1])
        }
        for b in result["results"]["bindings"]
    ]
    return sub_offices

def fetch_office_holders(qid, batch_size=2000, limit=None):
    """
    Directly fetch all holders of this office, regardless of country, logic simplified.
    """
    q_count = f"""
    SELECT (COUNT(*) AS ?count) WHERE {{
      ?entity wdt:P39 wd:{qid}.
    }}
    """
    total_count = get_count_for_query(q_count)
    if limit is not None and total_count is not None and total_count > limit:
        return [], True, total_count
    all_holders = []
    q = f"""
    SELECT ?entity ?entityLabel ?startDate ?endDate ?country ?countryLabel WHERE {{
      ?entity wdt:P39 wd:{qid}.
      OPTIONAL {{ 
        ?entity p:P39 ?stmt. 
        ?stmt ps:P39 wd:{qid}.
        OPTIONAL {{ ?stmt pq:P580 ?startDate. }}
        OPTIONAL {{ ?stmt pq:P582 ?endDate. }}
      }}
      OPTIONAL {{ ?entity wdt:P27 ?country. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC(?startDate)
    LIMIT {batch_size}
    """
    fetch_result = fetch_and_validate(q, qid, "", "holders", batch_size, years_key="holders")
    if isinstance(fetch_result, tuple):
        _, result = fetch_result
        if result.get("skipped"):
            return [], True, total_count
        return [], False, total_count
    result = fetch_result
    bindings = result.get("results", {}).get("bindings", []) if result else []
    if not bindings:
        return [], False, total_count
    for b in bindings:
        try:
            ent = b["entity"]["value"].split("/")[-1]
            label = b.get("entityLabel", {}).get("value", ent)
            start_date = b.get("startDate", {}).get("value", "unknown")
            end_date = b.get("endDate", {}).get("value", "unknown")
            country = b.get("country", {}).get("value", "").split("/")[-1] if "country" in b else ""
            country_label = b.get("countryLabel", {}).get("value", country)
            all_holders.append([label, ent, country_label, country, start_date, end_date])
        except Exception as e:
            continue
    return all_holders, False, total_count

def build_year_map(holders):
    year_map = {}
    current_year = datetime.now().year
    for i, h in enumerate(holders):
        if len(h) < 5:
            continue
        label, ent, country_label, country, start_date = h[:5]
        info = [label, ent, country_label, country]
        if start_date != "unknown":
            try:
                year = int(start_date[:4])
                if 1800 <= year <= current_year:
                    year_map.setdefault(str(year), []).append(info)
                else:
                    year_map.setdefault("invalid_year", []).append(info)
            except (ValueError, IndexError) as e:
                year_map.setdefault("unknown", []).append(info)
        else:
            year_map.setdefault("unknown", []).append(info)
    sorted_map = dict(sorted(
        year_map.items(), 
        key=lambda kv: (kv[0] == "unknown", -int(kv[0]) if kv[0].isdigit() else 0)
    ))
    return sorted_map

def fetch_office_holders_for_sub(sub_qid, parent_qid, sub_label, limit=2000, view_12m=None):
    try:
        holders, skipped, total_count = fetch_office_holders(sub_qid, batch_size=limit, limit=limit)
        if skipped:
            ret = {
                "qid": sub_qid,
                "parent_qid": parent_qid,
                "years": {},
                "skipped": True,
                "total_count": total_count
            }
            if view_12m is not None:
                ret["views_12m"] = view_12m
            return (sub_label, ret)
        year_map = build_year_map(holders)
        ret = {
            "qid": sub_qid,
            "parent_qid": parent_qid,
            "years": year_map
        }
        if view_12m is not None:
            ret["views_12m"] = view_12m
        return (sub_label, ret)
    except Exception as e:
        ret = {
            "qid": sub_qid,
            "parent_qid": parent_qid,
            "years": {}
        }
        if view_12m is not None:
            ret["views_12m"] = view_12m
        return (sub_label, ret)

def process_office(office, limit, parent_qid=None, max_workers=INNER_WORKERS):
    try:
        if isinstance(office, dict):
            qid = office.get("qid")
            label = office.get("label", "Main Office")
            view_12m = office.get("views_12m", None)
        else:
            qid = office
            label = "Main Office"
            view_12m = None
        sub_offices = get_sub_offices(qid, limit)
        sub_offices.insert(0, {"qid": qid, "label": label})
        submap = {qid: list({s["qid"] for s in sub_offices})}
        facts = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(fetch_office_holders_for_sub, s["qid"], qid, s["label"], limit, view_12m=view_12m): s 
                for s in sub_offices
            }
            for f in as_completed(futures, timeout=180):  # 3 minute timeout
                try:
                    result = f.result()
                    if isinstance(result, tuple) and len(result) == 2:
                        sub_label, info = result
                        facts[sub_label] = info
                except Exception as e:
                    continue
        return submap, facts
    except Exception as e:
        return {}, {}

# ==================== river strategy ====================

def get_river_countries(qid, label, limit, view_12m=None):
    """
    Get the countries traversed by a river, supports paging, keeps the view_12m field
    """
    query_template = """
    SELECT DISTINCT ?country ?countryLabel WHERE {{
      wd:{qid} wdt:P17 ?country.   # Countries traversed by river
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit} OFFSET {offset}
    """

    bindings = run_paged_query(qid, query_template, limit, "river_countries")

    countries = [
        {
            "qid": b["country"]["value"].split("/")[-1],
            "label": b.get("countryLabel", {}).get("value", b["country"]["value"].split("/")[-1]),
            "view_12m": view_12m
        }
        for b in bindings
    ]
    return countries

def get_sub_rivers(qid, label, limit, view_12m=None):
    """
    Get river's tributaries (tributary, P974), supports paging and keeps the view_12m field
    """
    query_template = """
    SELECT DISTINCT ?sub ?subLabel ?country ?countryLabel ?length WHERE {{
      ?sub wdt:P31 wd:Q4022.      # Must be river
      ?sub wdt:P974 wd:{qid}.     # Is a tributary of this river
      OPTIONAL {{ ?sub wdt:P17 ?country. }}
      OPTIONAL {{ ?sub wdt:P2043 ?length. }}  # Length
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit} OFFSET {offset}
    """

    bindings = run_paged_query(qid, query_template, limit, "sub_rivers")

    sub_list = [
        {
            "qid": b["sub"]["value"].split("/")[-1],
            "label": b.get("subLabel", {}).get("value", b["sub"]["value"].split("/")[-1]),
            "country": b.get("countryLabel", {}).get("value"),
            "length": b.get("length", {}).get("value"),
            "view_12m": view_12m
        }
        for b in bindings
    ]
    return sub_list


def process_river(river_entity, limit):
    """
    Fetch river and all its tributaries, as well as the countries they pass through
    """
    label, qid = river_entity["label"], river_entity["qid"]
    view_12m = river_entity.get("views_12m", None)
    facts = {}

    # First fetch main river's countries
    main_countries = get_river_countries(qid, label, limit, view_12m=view_12m)

    # Include itself
    all_rivers = [{"qid": qid, "label": label, "view_12m": view_12m}]

    # Fetch tributaries concurrently
    with ThreadPoolExecutor(max_workers=INNER_WORKERS) as pool:
        future_to_river = {
            pool.submit(get_sub_rivers, r["qid"], r["label"], limit, view_12m=r.get("view_12m", view_12m)): r
            for r in all_rivers
        }

        for f in as_completed(future_to_river):
            sub_rivers = f.result()
            if not sub_rivers:
                continue
            r = future_to_river[f]
            facts[r["label"]] = {
                "label": r["label"],
                "qid": r["qid"],
                "countries": main_countries,   # Main river's countries
                "tributaries": sub_rivers,     # Tributaries and the countries they pass through
                "views_12m": r.get("view_12m", None)
            }

    return {}, facts

# ==================== Main processing function ====================
def process_entity(entity, file_name, output_dir, limit, retry=3, timeout=30,  top_k=2000, outer_workers=OUTER_WORKERS, inner_workers=INNER_WORKERS, view_num = 1):
    FACTS_FILE = Path(output_dir) / f'{entity}_structured_facts.json'
    MAP_FILE = Path(output_dir) / f'{entity}_sub_mapping.json'

    with open(file_name, "r", encoding="utf-8") as f:
        entities = json.load(f)[entity]

    print(f"{entity} : entity count before filtering: {len(entities)}")
    top_entities = sorted(entities, key=lambda x: -x["views_12m"])[:top_k]
    top_entities = [e for e in top_entities if e.get("views_12m", 0) >= view_num]
    print(f"{entity} : entity count after topk and view_num filtering: {len(top_entities)}")

    all_mapping = {}
    all_facts = {}

    processor_map = {
        "award": process_award,
        "region": process_country,
        "language": process_language,
        "river" : process_river,
        "office": process_office,
    }
    
    processor = processor_map.get(entity)
    if not processor:
        print(f"❌ No processing strategy found for {entity}")
        return

    # Increase outer concurrency
    with ThreadPoolExecutor(max_workers=outer_workers) as outer_pool:
        futures = {outer_pool.submit(processor, e, limit): e["label"] for e in top_entities}
        for idx, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Processing {entity}")):
            try:
                mapping, facts = f.result()
                all_mapping.update(mapping)
                all_facts.update(facts)
            except Exception as e:
                continue


    FACTS_FILE.write_text(json.dumps(all_facts, ensure_ascii=False, indent=2))
    # MAP_FILE.write_text(json.dumps(all_mapping, ensure_ascii=False, indent=2))
    # print(f"\n✅ {entity} processed and saved to {FACTS_FILE} and {MAP_FILE}")
    print(f"\n✅ {entity} processed and saved to {FACTS_FILE} and {MAP_FILE}")


def main():
    args = parse_arguments()
    global OUTER_WORKERS, INNER_WORKERS
    OUTER_WORKERS = args.outer_workers
    INNER_WORKERS = args.inner_workers

    print("Current command line arguments (args):")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # party, mountain, religion, writer removed
    ENTITY_CONFIG = {
        "award":    {"file": os.path.join(args.entity_popularity_dir, "award_popularity.json")},
        "region":  {"file": os.path.join(args.entity_popularity_dir, "region_popularity.json")},
        "language": {"file": os.path.join(args.entity_popularity_dir, "language_popularity.json")},
        "office":   {"file": os.path.join(args.entity_popularity_dir, "office_popularity.json")},
        "river":    {"file": os.path.join(args.entity_popularity_dir, "river_popularity.json")},
    }
    
    print(f"📁 Entity popularity directory: {args.entity_popularity_dir}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"📁 Created output directory: {args.output_dir}")

    if args.entity != 'all':
        specified_entities = [e.strip() for e in args.entity.split(',')]
        invalid_entities = [e for e in specified_entities if e not in ENTITY_CONFIG]
        if invalid_entities:
            print(f"❌ Unsupported entity types: {', '.join(invalid_entities)}")
            print(f"Supported entity types: {', '.join(ENTITY_CONFIG.keys())}")
            sys.exit(1)
        ENTITY_CONFIG = {k: v for k, v in ENTITY_CONFIG.items() if k in specified_entities}
        print(f"📋 Will process these entity types: {', '.join(ENTITY_CONFIG.keys())}")


    print(f"⚙️  Config information:")
    print(f"   - Entity popularity directory: {args.entity_popularity_dir}")
    print(f"   - Output directory: {args.output_dir}")
    print(f"   - Entity types to process: {', '.join(ENTITY_CONFIG.keys())}")
    print(f"   - Query limit: {args.limit}")
    print(f"   - Retry count: {args.retry}")
    print(f"   - Query timeout: {args.timeout}")
    print(f"   - Outer concurrency: {OUTER_WORKERS}")
    print(f"   - Inner concurrency: {INNER_WORKERS}")
    for entity, conf in ENTITY_CONFIG.items():
        file_name = conf["file"]
        if not os.path.exists(file_name):
            print(f"❌ {file_name} not found, skipping {entity}")
            continue
        print(f"\n{'='*20} START processing entity: {entity.upper()} {'='*20}")
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
                entity_count = len(data[entity])
                print(f"📊 Found {entity_count} {entity} entities")
            process_entity(entity, file_name, args.output_dir, args.limit, retry=args.retry, timeout=args.timeout, top_k = args.top_k, outer_workers=OUTER_WORKERS, inner_workers=INNER_WORKERS, view_num=args.view_num)
        except Exception as e:
            print(f"❌ Error occurred while processing {entity}: {e}")
            continue

if __name__ == "__main__":
    main()