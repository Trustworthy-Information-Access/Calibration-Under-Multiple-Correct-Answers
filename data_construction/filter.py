import argparse
import os
import json

def filter_china_related(facts, data_key, entity):
    """
    Filter out all facts entries related to China.
    Criterion: Any value under facts[data_key], including nested lists/tuples etc., containing 'china' or '中国' or 'chinese' (case-insensitive).
    """
    def is_china_related(val):
        if isinstance(val, str):
            # return ("china" in val.lower()) or ("中国" in val) or ("chinese" in val.lower())
            return ("china" in val.lower()) or ("中国" in val) or ("chinese" in val.lower())
        elif isinstance(val, (list, tuple)):
            return any(is_china_related(x) for x in val)
        elif isinstance(val, dict):
            return any(is_china_related(x) for x in val.values())
        return False

    def any_china_in_datakey_value(v, data_key):
        value = v.get(data_key, None)
        if value is None:
            return False
        # value can be dict, list, tuple, str, etc.
        if isinstance(value, dict):
            # Check every key and every value
            for k1, v1 in value.items():
                if is_china_related(k1) or is_china_related(v1):
                    return True
            return False
        elif isinstance(value, (list, tuple)):
            return any(is_china_related(x) for x in value)
        else:
            return is_china_related(value)

    filtered_facts = {}
    for k, v in facts.items():
        # If any value under data_key or k itself is china-related, filter it out
        if is_china_related(k):
            continue
        if not any_china_in_datakey_value(v, data_key):
            filtered_facts[k] = v
    print(f"[{entity}] --filter_china_related count after filter: {len(filtered_facts)}")
    
    return filtered_facts

invalid_language_labels = ["Yiddish"]

def filter_specific_item_lang(facts):
    filtered_facts = {}
    for k, v in facts.items():
        if any(label in k for label in invalid_language_labels):
            continue
        filtered_facts[k] = v
    print(f"[language] --filter_specific_item count after filter: {len(filtered_facts)}")
    return filtered_facts

invalid_award_labels = [
    # team/organization etc.
    "team", "group", "organization", "institution", "association", "company",
    # # work/project related
    # "film", "movie", "book", "album", "song", "track",
    # "project", "work", "performance", "production",
    # collection/multi-person
    "winners", "recipients", "authors", "directors", "producers", "artists",
    "player"
]

def filter_specific_item_award(facts):
    filtered_facts = {}
    for k, v in facts.items():
        if any(label.lower() in k.lower() for label in invalid_language_labels):
            continue
        filtered_facts[k] = v
    print(f"[language] --filter_specific_item count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_no_english_country(facts):
    """
    Filter out countries where any division has a non-English label.
    Only keep countries where all 'divisions'['name'] fields are in English.
    """
    import re

    def is_english(s):
        # Only allow English letters, digits, space, and common punctuation.
        # e.g. "Aosta Valley", "Trentino-South Tyrol", "Lombardy"
        # Disallow non-English chars like Chinese, Arabic, Russian etc.
        if not isinstance(s, str):
            return False
        # Allow English, digit, space, -, etc.
        return re.fullmatch(r"[A-Za-z0-9\s\-\',\(\)\.]+", s) is not None

    filtered_facts = {}
    for country, v in facts.items():
        divisions = v.get("divisions", [])
        all_english = True
        for div in divisions:
            name = div.get("name", "")
            if not is_english(name):
                # print(country)
                all_english = False
                break
        if all_english:
            filtered_facts[country] = v
    print(f"[country] --filter_no_english_country count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_multi_qid(facts, data_key, entity):
    """
    Extract the ["qid"] field from v directly, remove duplicate qids, and only keep the first record for each qid.
    """
    filtered_facts = {}
    seen_qids = set()
    for k, v in facts.items():
        qid = v.get("qid", None)
        if qid is not None and qid not in seen_qids:
            seen_qids.add(qid)
            filtered_facts[k] = v
    print(f"[{entity}] --filter_multi_qid count after filter: {len(filtered_facts)}")
    return filtered_facts


def filter_no_value(facts, data_key, entity):
    """
    If "unknown" exists, keep only the maximal consecutive year range (skip this label if <2 years).
    All other logic same as before.
    """
    def get_max_consecutive_years(years_dict):
        # Extract all numeric years
        years = [int(y) for y in years_dict.keys() if y.isdigit()]
        if not years:
            return {}
        years = sorted(years)
        # Find max consecutive subsequence
        max_seq = []
        curr_seq = []
        for i, y in enumerate(years):
            if not curr_seq:
                curr_seq = [y]
            elif y == curr_seq[-1] + 1:
                curr_seq.append(y)
            else:
                if len(curr_seq) > len(max_seq):
                    max_seq = curr_seq
                curr_seq = [y]
        if len(curr_seq) > len(max_seq):
            max_seq = curr_seq
        # If <2 in max consecutive length, return empty
        if len(max_seq) < 2:
            return {}
        # Build new years dict
        return {str(y): years_dict[str(y)] for y in max_seq if str(y) in years_dict}

    filtered_facts = {}
    seen_keys = set()
    for k, v in facts.items():
        if isinstance(v, dict) and "years" in v:
            if not v["years"]:
                continue

        value = v.get(data_key)
        if value is None or value == [] or value == {} or value == "":
            continue

        if isinstance(value, dict):
            # handle "unknown"
            if "unknown" in value.keys() and entity != "award":
                # Keep only maximal consecutive years
                max_consecutive = get_max_consecutive_years(value)
                if not max_consecutive:
                    continue
                v[data_key] = max_consecutive
            elif entity == "award":
                # If the only key is "unknown" or "invalid_year", skip this entry
                if set(value.keys()) == {"unknown"} or set(value.keys()) == {"invalid_year"}:
                    continue
            # Only keep items whose key is all digits
            # if entity != "award":
            keys_to_remove = [key for key in value.keys() if not key.isdigit()]
            for key in keys_to_remove:
                seen_keys.add(key)
                value.pop(key)

        filtered_facts[k] = v

    if seen_keys:
        print(seen_keys)
    print(f"[{entity}] --filter_no_value count after filter: {len(filtered_facts)}")

    return filtered_facts

def get_all_type(facts, entity, output_file="/mnt/public/gpfs-jd/data/lh/wyh/Multi_Answer_Confidence/generate_data/new_code/entity_structure_map_topk2000_view10000_filter_new_final_opt/type_stat.txt"):
    """
    Count all 'type' fields among facts and write out unique types and their counts to a file.
    """
    type_set = set()
    for k, v in facts.items():
        # handle type field if exists
        if "type" in v:
            type_val = v["type"]
            if isinstance(type_val, list):
                for t in type_val:
                    type_set.add(t)
            else:
                type_set.add(type_val)
        # handle type under each division (for country-structured data)
        if "divisions" in v and isinstance(v["divisions"], list):
            for div in v["divisions"]:
                if isinstance(div, dict) and "type" in div:
                    type_val = div["type"]
                    if isinstance(type_val, list):
                        for t in type_val:
                            type_set.add(t)
                    else:
                        type_set.add(type_val)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"[{entity}] Number of types: {len(type_set)}\n")
        f.write("All types:\n")
        for t in sorted(type_set):
            f.write(f"{t}\n")
    print(f"[{entity}] Number of types: {len(type_set)}, written to file: {output_file}")
    return type_set

def filter_wiki_web(facts, entity):
    """
    Strict filter: As long as any person's name under 'years' is "http://www.wikidata.org/.well-known/genid/...", drop the whole label/fact entry.
    """
    def is_wiki_web(val):
        return isinstance(val, str) and val.startswith("http://www.wikidata.org/.well-known/genid/")

    filtered_facts = {}
    for k, v in facts.items():
        # Check years field
        if "years" in v and isinstance(v["years"], dict):
            found_abnormal = False
            for year, people_list in v["years"].items():
                # New: If the year itself is an http link, also drop
                if isinstance(year, str) and year.startswith("http"):
                    found_abnormal = True
                    break
                for person in people_list:
                    # 'person' is generally a list, first element is name
                    if isinstance(person, list) and len(person) > 0:
                        if is_wiki_web(person[0]):
                            found_abnormal = True
                            break
                if found_abnormal:
                    break
            if found_abnormal:
                continue  # If any abnormal value, skip this label entirely
        filtered_facts[k] = v
    
    print(f"[{entity}] --filter_wiki_web count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_label_qid(facts, data_key, entity):
    """
    Strict filter: If the label itself or any label under data_key is a string starting with Q followed by all digits, remove the entire entry.
    """
    filtered_facts = {}
    for label, v in facts.items():
        # filter out entries whose label is of form Qxxxx...
        if label.startswith("Q") and label[1:].isdigit():
            continue
        abnormal = False
        # Process data_key field content
        if data_key in v:
            value = v[data_key]
            # If list, check items for Qxxxx... labels
            if isinstance(value, list):
                for item in value:
                    # item can be list/tuple/dict/str
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        if isinstance(item[0], str) and item[0].startswith("Q") and item[0][1:].isdigit():
                            abnormal = True
                            break
                    elif isinstance(item, (dict)) and len(item) > 0:
                        if "label" in item:
                            if isinstance(item["label"], str) and item["label"].startswith("Q") and item["label"][1:].isdigit():
                                abnormal = True
                                break
                        else:
                            if isinstance(item["name"], str) and item["name"].startswith("Q") and item["name"][1:].isdigit():
                                abnormal = True
                                break
            # If dict, check if any value's first string is a qid
            elif isinstance(value, dict):
                for k2, v2 in value.items():
                    # v2 usually is list, take the first element
                    if isinstance(v2, (list, tuple)) and len(v2) > 0:
                        first_elem = v2[0]
                        if isinstance(first_elem, str) and first_elem.startswith("Q") and first_elem[1:].isdigit():
                            abnormal = True
                            break
           
        if abnormal:
            continue  # If any abnormal value found, skip this label
        filtered_facts[label] = v
    print(f"[{entity}] --filter_label_qid count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_one_country(facts, entity):
    """
    Only keep facts where the specified field (item[2] for office, item[3] for party in the per-year value) has only one unique value.
    """
    filtered_facts = {}
    for k, v in facts.items():
        years = v.get("years", {})
        country_set = set()
        for year, year_values in years.items():
            if isinstance(year_values, list):
                for item in year_values:
                    # Only handle lists/tuples with at least 5 elements (like ["name", "qid", "Q5", "country_name", "country_qid"])
                    if isinstance(item, (list, tuple)) :
                        if entity == "office" and len(item) >= 3 and not item[2]:
                            country_set.add(item[2])
                        elif entity == "party" and len(item) >= 4 and not item[3]:
                            country_set.add(item[3])
        if len(country_set) == 1:
            filtered_facts[k] = v
        
    print(f"[{entity}] --filter_one_country count after filter: {len(filtered_facts)}")
    return filtered_facts

# INSERT_YOUR_REWRITE_HERE
def filter_count_eqN(facts, data_key, n):
    """
    Only keep facts whose data_key field has exactly n items.
    :param facts: dict, input facts
    :param data_key: str, field name to count (e.g. "years", "divisions", etc.)
    :param n: int, minimum number of items
    :return: dict, filtered facts
    """
    filtered_facts = {}
    for k, v in facts.items():
        data_field = v.get(data_key, {})
        # Only count valid keys (if dict, count keys; if list, count length; else 0)
        if isinstance(data_field, dict):
            key_count = len(data_field.keys())
        elif isinstance(data_field, list):
            key_count = len(data_field)
        else:
            key_count = 0
        if key_count == n:
            filtered_facts[k] = v
    
    print(f"[{entity}] --filter_count_geN count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_count_geN(facts, data_key, n):
    """
    Only keep facts whose data_key field has >= n items.
    :param facts: dict, input facts
    :param data_key: str, field name to count (e.g. "years", "divisions", etc.)
    :param n: int, minimum number of items
    :return: dict, filtered facts
    """
    filtered_facts = {}
    for k, v in facts.items():
        data_field = v.get(data_key, {})
        # Only count valid keys (if dict, count keys; if list, count length; else 0)
        if isinstance(data_field, dict):
            key_count = len(data_field.keys())
        elif isinstance(data_field, list):
            key_count = len(data_field)
        else:
            key_count = 0
        if key_count >= n:
            filtered_facts[k] = v
    
    print(f"[{entity}] --filter_count_geN count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_count_leN(facts, data_key, n):
    """
    Only keep facts whose data_key field has <= n items.
    :param facts: dict, input facts
    :param data_key: str, field name to count (e.g. "years", "divisions", etc.)
    :param n: int, maximum number of items
    :return: dict, filtered facts
    """
    filtered_facts = {}
    for k, v in facts.items():
        data_field = v.get(data_key, {})
        # Only count valid keys (if dict, count keys; if list, count length; else 0)
        if isinstance(data_field, dict):
            key_count = len(data_field.keys())
        elif isinstance(data_field, list):
            key_count = len(data_field)
        else:
            key_count = 0
        if key_count <= n:
            filtered_facts[k] = v
    
    print(f"[{entity}] --filter_count_geN count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_one_person_per_year(facts):
    """
    Only keep facts where every year has exactly one unique awardee (deduplicated by name).
    If any year has multiple different winners, skip the whole label.
    """
    filtered_facts = {}
    for label, v in facts.items():
        years = v.get("years", {})
        valid = True
        for year, year_values in years.items():
            # year_values should be a list representing all awardees that year
            if not isinstance(year_values, list) or len(year_values) == 0:
                valid = False
                break
            # For each year, deduplicate by item[0] (name)
            name_set = set()
            for item in year_values:
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    name_set.add(item[0])
                else:
                    valid = False
                    break
            if not valid:
                break
            if len(name_set) != 1:
                valid = False
                break
        if valid:
            filtered_facts[label] = v
    print(f"[award] --filter_one_person_per_year count after filter: {len(filtered_facts)}")
    return filtered_facts

def filter_one_country_per_lang(facts):
    filtered_facts = {}
    country_count_stat = {}
    for label, v in facts.items():
        countries = v.get("countries", [])
        # countries is expected as a list, each element also a list, first element is country_name
        country_names = set()
        for item in countries:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                country_names.add(item[0])
        num_countries = len(country_names)
        country_count_stat[num_countries] = country_count_stat.get(num_countries, 0) + 1
        if num_countries == 1:
            filtered_facts[label] = v
    print(f"[award] --filter_one_country_per_lang count after filter: {len(filtered_facts)}")
    # print("Distribution of country_name count:")
    # for k in sorted(country_count_stat.keys()):
    #     print(f"  {k} countries: {country_count_stat[k]} records")
    return filtered_facts

def print_entity_data_num(facts, entity):
    if entity == "party":
        years_count_stat = {}
        for label, v in facts.items():
            years = v.get("years", {})
            num_years = len(years)
            years_count_stat[num_years] = years_count_stat.get(num_years, 0) + 1
        print(f"[{entity}] Number of years : Number of entries")
        for k in sorted(years_count_stat.keys()):
            print(f"{k}: {years_count_stat[k]}")
    return

def get_complete_father(facts, submap):
    # Read all qids
    all_qids = set(v.get("qid") for v in facts.values() if v.get("qid") is not None)

    # Find all "complete parent classes" (i.e. parent and all children are in facts)
    complete_parent_qids = set()
    for k, v in submap.items():
        if not v:
            continue
        # parent and all children exist in facts
        if all(x in all_qids for x in v) and k in all_qids:
            complete_parent_qids.add(k)

    # Only remove incomplete parent; incomplete parent's children should still be kept.
    # Build new facts, only remove incomplete parent classes.
    new_facts = {}
    for k, v in facts.items():
        qid = v.get("qid")
        # Only keep if not an incomplete parent (i.e., remove if parent but not complete; keep others)
        if qid is None:
            continue
        # Check whether is parent
        is_parent = qid in submap
        if is_parent and qid not in complete_parent_qids:
            continue  # skip incomplete parent
        new_facts[k] = v

    # Process submap, only keep complete parents and their children (children must be in new_facts too)
    if submap:
        filtered_qids = set(v.get("qid") for v in new_facts.values() if v.get("qid") is not None)
        filtered_submap = {}
        # Only keep "complete parent classes"
        for k, v in submap.items():
            if not v:
                continue
            if k in complete_parent_qids:
                # Only keep children also in filtered_qids
                filtered_v = [x for x in v if x in filtered_qids]
                if filtered_v:
                    filtered_submap[k] = filtered_v
        submap = filtered_submap

    print(f"[{entity}] --filter_complete_father count after filter: {len(new_facts)}")
    return new_facts, submap

ENTITY_CONFIG = {
    "award": "years",
    "office": "years",
    "region": "divisions",
    "language": "countries",
    "river" : "countries"
}


def filter_consecutive_years(years: dict):
    """
    Given a years dict (keys are years, values are all_entities), only keep the longest consecutive range (length >=2), return new years dict.
    """
    def find_longest_consecutive_subseq(year_list):
        # Only keep numeric years and skip 2024 and 2025
        year_keys = list(years.keys())
        year_keys = [k for k in year_keys if k not in ("2024", "2025") and k.isdigit() and int(k) >= 1800 ]
        year_keys = [int(k) for k in year_keys]
        if not year_keys:
            return []
        year_keys = sorted(set(year_keys))
        max_seq = []
        curr_seq = []
        for y in year_keys:
            if not curr_seq:
                curr_seq = [y]
            elif y == curr_seq[-1] + 1:
                curr_seq.append(y)
            else:
                if len(curr_seq) > len(max_seq):
                    max_seq = curr_seq
                curr_seq = [y]
        if len(curr_seq) > len(max_seq):
            max_seq = curr_seq
        return [str(y) for y in max_seq] if len(max_seq) >= 2 else []

    year_keys = []
    for y in years.keys():
        if str(y).isdigit():
            if y in ("2024", "2025"):
                continue
            year_keys.append(y)
        elif str(y).lower() == "current":
            continue
    longest_seq = find_longest_consecutive_subseq(year_keys)
    if not longest_seq:
        return {}
    # Only keep the longest consecutive range of years
    new_years = {yk: years[yk] for yk in longest_seq}
    return new_years

def filter_has_consecutive_years(facts):
    """
    Filter out facts without a consecutive range of at least 2 years.
    """

    new_facts = {}
    for k, v in facts.items():
        years = v.get("years", {})
        # Use filter_consecutive_years to filter
        filtered_years = filter_consecutive_years(years)
        if len(filtered_years) >= 2:
            # Only keep the longest consecutive years
            v["years"] = {yk: years[yk] for yk in filtered_years}
            new_facts[k] = v
    print(f"[years] --filter_has_consecutive_years count after filter: {len(new_facts)}")
    return new_facts

def get_valid_submap(facts, submap):
    # Only keep qids existing in filtered facts
    filtered_qids = set(v.get("qid") for v in facts.values() if v.get("qid") is not None)
    filtered_submap = {}
    for k, v in submap.items():
        if k in filtered_qids:
            # Only keep child ids also in filtered_qids
            filtered_v = [x for x in v if x in filtered_qids]
            if filtered_v:
                filtered_submap[k] = filtered_v
    return facts, filtered_submap

def filter_no_area_country(facts):
    """
    Filter out countries where any division lacks an area.
    """
    new_facts = {}
    for country, v in facts.items():
        divisions = v.get("divisions", [])
        # If any division is missing area, skip this country
        if any(d.get("area") is None for d in divisions):
            continue
        new_facts[country] = v
    print(f"[country] --filter_no_area_country count after filter: {len(new_facts)}")
    return new_facts

def filter_all_entity_facts(entity, input_dir, output_dir):
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if entity not in ENTITY_CONFIG:
        print(f"Entity type {entity} not in supported list: {list(ENTITY_CONFIG.keys())}")
        return
    file = f"{entity}_structured_facts.json"
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)
    submap_file = f"{entity}_sub_mapping.json"
    submap_input_path = os.path.join(input_dir, submap_file)
    submap_output_path = os.path.join(output_dir, submap_file)

    if not os.path.exists(input_path):
        print(f"[{entity}] Skip missing file: {input_path}")
        return
    with open(input_path, 'r', encoding='utf-8') as fin:
        facts = json.load(fin)

    print(f"\nEntity: {entity}")
    print(f"[{entity}] facts count before filter: {len(facts)}")

    # Filter out facts without valid years
    facts = filter_no_value(facts, ENTITY_CONFIG[entity], entity)
    facts = filter_label_qid(facts, ENTITY_CONFIG[entity], entity)
    facts = filter_china_related(facts, ENTITY_CONFIG[entity], entity)
    facts = filter_multi_qid(facts, ENTITY_CONFIG[entity], entity)

    if entity == "award":
        facts = filter_one_person_per_year(facts)
        facts = filter_specific_item_award(facts)
        facts = filter_has_consecutive_years(facts)
        facts = filter_count_geN(facts, ENTITY_CONFIG[entity], 3)
        # facts = filter_count_leN(facts, ENTITY_CONFIG[entity], 10)
    if entity == "office":
        # facts = filter_one_country(facts, entity)
        facts = filter_one_person_per_year(facts)
        # facts = filter_count_eqN(facts, "years", 1)
        facts = filter_count_geN(facts, ENTITY_CONFIG[entity], 10)
        # facts = filter_count_leN(facts, ENTITY_CONFIG[entity],15)
        # facts = filter_count_leN(facts, ENTITY_CONFIG[entity], 6)
    if entity == "language":
        # facts = filter_specific_item_lang(facts)
        one_facts = filter_count_eqN(facts, ENTITY_CONFIG[entity], 1)
        facts = filter_count_geN(facts, ENTITY_CONFIG[entity], 2)
        # facts = filter_count_leN(facts, ENTITY_CONFIG[entity], 8)
        facts = filter_count_leN(facts, ENTITY_CONFIG[entity], 6)

        # Randomly sample 600 one_facts and merge into facts
        one_facts_items = list(one_facts.items())
        if len(one_facts_items) > 600:
            sampled_one_facts = dict(random.sample(one_facts_items, 600))
        else:
            sampled_one_facts = one_facts
        facts.update(sampled_one_facts)
        # facts_back = facts
        # facts = filter_one_country_per_lang(facts)
    if entity == "region":
        facts = filter_no_area_country(facts)
        facts = filter_count_geN(facts, ENTITY_CONFIG[entity], 2)
    if entity == "river":
        one_facts = filter_count_eqN(facts, ENTITY_CONFIG[entity], 1)
        facts = filter_count_geN(facts, ENTITY_CONFIG[entity], 2)
        # facts = filter_count_leN(facts, ENTITY_CONFIG[entity], 6)
        facts = filter_count_leN(facts, ENTITY_CONFIG[entity], 6)
        one_facts_items = list(one_facts.items())
        if len(one_facts_items) > 600:
            sampled_one_facts = dict(random.sample(one_facts_items, 600))
        else:
            sampled_one_facts = one_facts
        facts.update(sampled_one_facts)

    # facts = filter_has_subclass(facts, entity)
  
    # Read submap
    if os.path.exists(submap_input_path):
        with open(submap_input_path, 'r', encoding='utf-8') as fsub:
            submap = json.load(fsub)
    else:
        submap = {}
    
    # if entity not in [""]:
    #     facts, submap = get_complete_father(facts, submap)
    # else:
    #     # print(len(facts))
    # facts, submap = get_valid_submap(facts, submap)
   
    print(f"[{entity}] count after filter: {len(facts)}")

    parent_count = 0
    child_count = 0
    for k, v in facts.items():
        qid = v.get("qid")
        parent_qid = v.get("parent_qid")
        if qid is not None and parent_qid is not None:
            if qid == parent_qid:
                parent_count += 1
            else:
                child_count += 1
    print(f"parent count: {parent_count}, child count: {child_count}")
    # Save filtered facts
    data = facts
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)
    # print(f"[{entity}] Output file path: {output_path} {submap_output_path}")
    print(f"[{entity}] Output file path: {output_path}")

    # with open(submap_output_path, 'w', encoding='utf-8') as fsubout:
    #     json.dump(submap, fsubout, ensure_ascii=False, indent=2)
        
import time
import random
from typing import Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from tqdm import tqdm

def filter_has_subclass(facts: Dict, entity, max_workers: int = 16) -> Dict:
    """
    Only keep entries that have NO (direct/indirect) subclass.
    - Use Wikidata P279+ relation to check if there are any-level subclasses
    - If query fails/uncertain, drop the fact (strict: only keep those that definitely have no subclass)
    - Each fact entry must have a "qid" field
    - Supports concurrency with thread pool
    Parameters
    ----
    facts: Dict
        {label: {"qid": "Qxxxx", ...}, ...}
    max_workers: int
        concurrency thread count
    Returns
    ----
    Dict
        Filtered facts (only entries without subclasses)
    """
    import concurrent.futures

    def _make_session(total_retries: int = 5, backoff: float = 0.5) -> requests.Session:
        sess = requests.Session()
        retries = Retry(
            total=total_retries,
            read=total_retries,
            connect=total_retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"])
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        sess.headers.update({
            "User-Agent": "Wikidata-Subclass-Checker/1.1 (contact: your_email@example.com)"
        })
        return sess

    SESSION = _make_session()

    def _wdqs_has_subclasses(qid: str, timeout_sec: int = 15) -> Optional[bool]:
        """True: has subclass; False: has no subclass; None: query failed or uncertain"""
        url = "https://query.wikidata.org/sparql"
        query = f"""
        ASK {{
          ?x wdt:P279+ wd:{qid} .
        }}
        """
        try:
            resp = SESSION.get(url, params={"query": query, "format": "json"}, timeout=timeout_sec)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return bool(data.get("boolean")) if "boolean" in data else None
        except Exception as e:
            return None

    # task function
    def check_subclass(label_item):
        label, item = label_item
        qid = (item or {}).get("qid")
        if not qid:
            return (label, item, None)
        # slight random delay to avoid batch-bombing WDQS
        time.sleep(random.uniform(0, 0.2))
        has_sub = _wdqs_has_subclasses(qid)
        return (label, item, has_sub)

    filtered: Dict = {}
    cache: Dict[str, Optional[bool]] = {}

    items = list(facts.items())

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {}
        for label, item in items:
            qid = (item or {}).get("qid")
            if not qid:
                print(f"[{label}] missing qid, skipped.")
                continue
            if qid in cache:
                # direct use cache
                has_sub = cache[qid]
                if has_sub is False:
                    filtered[label] = item
                # print each fact's subclass status
                if has_sub is True:
                    print(f"[{label}] (qid={qid}) has subclass.")
                elif has_sub is False:
                    print(f"[{label}] (qid={qid}) no subclass.")
                else:
                    print(f"[{label}] (qid={qid}) query failed/network error.")
                continue
            # submit task
            future = executor.submit(check_subclass, (label, item))
            future_to_label[future] = label

        for fut in tqdm(concurrent.futures.as_completed(future_to_label), total=len(future_to_label), desc="Concurrent subclass check", ncols=80):
            label, item, has_sub = fut.result()
            qid = (item or {}).get("qid")
            if qid:
                cache[qid] = has_sub
            # print each fact's subclass status
            if has_sub is True:
                print(f"[{label}] (qid={qid}) has subclass.")
            elif has_sub is False:
                print(f"[{label}] (qid={qid}) no subclass.")
            else:
                print(f"[{label}] (qid={qid}) query failed/network error.")
            if has_sub is False:
                filtered[label] = item

    print(f"[{entity}] --filter_sub_class count after filter: {len(filtered)}")
    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter structured facts data")
    parser.add_argument("--entity", type=str, required=True, help="Entity type, e.g. writer/award/office/all etc.")
    parser.add_argument("--topk", type=str, default="2000", help="topk number (used in path)")
    parser.add_argument("--input_dir", type=str, required=False, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory")
    args = parser.parse_args()

    entity = args.entity
    topk = args.topk

    # If input/output dirs not provided, use defaults
    if args.input_dir:
        input_dir = args.input_dir
    if args.output_dir:
        output_dir = args.output_dir
    print(input_dir)
    print(output_dir)
   
    if entity == "all":
        for ent in ENTITY_CONFIG.keys():
            filter_all_entity_facts(ent, input_dir, output_dir)
    else:
        filter_all_entity_facts(entity, input_dir, output_dir)
