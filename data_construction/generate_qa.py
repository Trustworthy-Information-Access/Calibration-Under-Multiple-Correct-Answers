import json
import math
import random
import sys
import os
import argparse
from datetime import datetime
from tqdm import tqdm

random.seed(42)

# ==================== Configuration Area ====================

QA_COUNT = 500

# Control the sampling ratio of different labels in each entity category
ENTITY_FILTER_CONFIG = {
    "award": 7 / QA_COUNT,
    "language": 4 / QA_COUNT,
    "office": 4 / QA_COUNT,
    "country": 9 / QA_COUNT,
    "river": 4 / QA_COUNT
}

# Entity configuration and question templates
ENTITY_CONFIG = {
    "award": {
        "entity_type": "award",
        "question_template": {
            "single": "Name one person who received {label} in {start}.",
            "double": "Name one person who received {label} between {start} and {end}."
        }
    },
    "region": {
        "entity_type": "region",
        "question_template": {
            "single": "Name one first-level administrative division of {country_name}, whose area (in square kilometers) is between {area_min} and {area_max}"
        }
    },
    "language": {
        "entity_type": "language",
        "question_template": {
            "single": "Name one country where {label1} was predominantly spoken by native speakers.",
            "double": "Name one country where {label1} or {label2} were predominantly spoken by native speakers.",
            "triple": "Name one country where {label1}, {label2}, or {label3} were predominantly spoken by native speakers."
        }
    },
    "math": {
        "entity_type": "math",
        "question_template": "Name one {number_type} number between {low} and {high}."
    },
    "office": {
        "entity_type": "office",
        "question_template": {
            "single": "Name one person from who officially assumed the office of {label} in {start}.",
            "double": "Name one person from who officially assumed the office of {label} between {start} and {end}."
        }
    },
    "river": {
        "entity_type": "river",
        "question_template": {
            "single": "Name one country through which the {label1} flows.",
            "double": "Name one country through which the {label1} or the {label2} flows.",
            "triple": "Name one country through which the {label1}, {label2}, or {label3} flows."
        }
    }
}

MAX_TRIES_PER_ENTITY = 100
MAX_FAILS = 200000


# ==================== Utility Functions ====================

def compute_difficulty(span, avg_per_year, nonhuman_ratio):
    return round(
        0.4 * math.log1p(span) + 0.6 * (2 - min(avg_per_year, 2)) + 1.0 * nonhuman_ratio,
        3
    )

# ==================== Math Strategy ====================

def generate_primes(upto):
    sieve = [True] * (upto + 1)
    sieve[0:2] = [False, False]
    for num in range(2, int(upto ** 0.5) + 1):
        if sieve[num]:
            sieve[num * num:upto + 1:num] = [False] * len(range(num * num, upto + 1, num))
    return [i for i, val in enumerate(sieve) if val]

def generate_squares(upto):
    return [i * i for i in range(1, int(upto ** 0.5) + 1)]

def generate_cubes(upto):
    return [i ** 3 for i in range(1, int(upto ** (1/3)) + 2) if i ** 3 <= upto]

def generate_fibonacci(upto):
    fibs = [1, 1]
    while fibs[-1] + fibs[-2] <= upto:
        fibs.append(fibs[-1] + fibs[-2])
    return list(set(fibs))

def generate_triangular(upto):
    tris = []
    i = 1
    while True:
        t = i * (i + 1) // 2
        if t > upto:
            break
        tris.append(t)
        i += 1
    return tris

MATH_GENERATORS = {
    "prime": generate_primes,
    "square": generate_squares,
    "cube": generate_cubes,
    "fibonacci": generate_fibonacci,
    "triangular": generate_triangular,
}

def preprocess_math_sequences(upto=1000000):
    """Preprocess all math type sets to avoid repeatedly generating them"""
    sequences = {}
    for k, func in MATH_GENERATORS.items():
        sequences[k] = func(upto)
        sequences[k].sort()
    return sequences

# Global cache
MATH_SEQUENCES_CACHE = preprocess_math_sequences(1000000)

def generate_math(min_answers, max_answers, qa_count, output_dir):
    """Generate math entity questions"""
    output_dir = os.path.join(output_dir, 'math')
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_FILE = f"{output_dir}/qa_math_{min_answers}_{max_answers}_{qa_count}.jsonl"
    print(f"\n=== Generating for entity: math ===")
    
    type_list = list(MATH_SEQUENCES_CACHE.keys())
    type_num = len(type_list)
    per_type_count = qa_count // type_num

    all_candidates = []
    seen_ranges = {k: set() for k in type_list}
    max_attempts = MAX_FAILS

    pbar = tqdm(total=qa_count, desc="Generating math", unit="QA")

    for number_type in type_list:
        full_set = MATH_SEQUENCES_CACHE[number_type]
        this_type_count = per_type_count
        attempts = 0
        
        while len([c for c in all_candidates if c["meta"]["entity_label"] == number_type]) < this_type_count and attempts < max_attempts:
            if len(full_set) < min_answers:
                attempts += 1
                continue
            
            # Randomly select the number of answers in the interval
            target_count = min_answers
            if len(full_set) - target_count <= 0:
                attempts += 1
                continue

            start_idx = random.randint(0, len(full_set) - target_count)
            end_idx = start_idx + target_count - 1

            # Determine interval boundaries [low, high]
            if start_idx > 0:
                prev_val = full_set[start_idx - 1]
                curr_val = full_set[start_idx]
                low = random.randint(prev_val + 1, curr_val)
            else:
                attempts += 1
                continue
            
            if end_idx + 1 < len(full_set):
                next_val = full_set[end_idx + 1]
                curr_val = full_set[end_idx]
                high = random.randint(curr_val, next_val - 1)
            else:
                attempts += 1
                continue
            
            if low >= high:
                attempts += 1
                continue
            
            candidates = list(set([x for x in full_set if low <= x <= high]))
            if len(candidates) != target_count:
                attempts += 1
                continue
            
            question = ENTITY_CONFIG["math"]["question_template"].format(
                number_type=number_type, low=low, high=high
            )
            
            range_key = question
            if range_key not in seen_ranges[number_type]:
                seen_ranges[number_type].add(range_key)
                all_candidates.append({
                    "question": question,
                    "reference": candidates,
                    "answer_count": len(candidates),
                    "meta": {
                        "entity_label": number_type,
                        "entity_type": "math"
                    },
                    "difficulty": compute_difficulty(high - low, len(candidates), 0.0)
                })
                attempts = 0
                pbar.update(1)
            else:
                attempts += 1

    # Deduplicate
    unique_candidates = []
    seen = set()
    for q in all_candidates:
        key = json.dumps(q, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(q)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for item in unique_candidates[:qa_count]:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Finished math. {len(unique_candidates[:qa_count])} QA written to {OUTPUT_FILE}")


# ==================== Award Strategy ====================

def generate_award(facts, min_answers, max_answers, qa_count, question_template):
    entity = "award"
    candidates = []
    fail_counter = 0
    unique_keys = set()
    pbar = tqdm(total=qa_count, desc=f"Generating {entity}")

    label_data_counter = {}
    labels = list(facts.keys())

    while len(candidates) < qa_count:
        if fail_counter >= MAX_FAILS:
            print("❌ Too many failures. Exiting.")
            break

        label = random.choice(labels)

        max_label_data = int(qa_count * ENTITY_FILTER_CONFIG[entity])
        if label_data_counter.get(label, 0) >= max_label_data:
            fail_counter += 1
            continue

        if label.startswith("Q") and label[1:].isdigit():
            fail_counter += 1
            continue

        # Handle year
        years = {}
        for year, people in facts[label].items():
            if year == "current":
                current_year = str(datetime.now().year)
                years.setdefault(current_year, []).extend(people)
            elif year.isdigit():
                years.setdefault(year, []).extend(people)
            else:
                continue

        if not years:
            fail_counter += 1
            continue

        year_keys = sorted([k for k in years.keys() if k.isdigit() and int(k) >= 1800], key=int)
        year_keys = [k for k in year_keys if k not in ("2024", "2025")]

        if not year_keys:
            fail_counter += 1
            continue

        min_span = 1
        max_span = len(year_keys)
        span = random.randint(min_span, max_span)
        max_start_idx = len(year_keys) - span
        
        if max_start_idx < 0:
            fail_counter += 1
            continue
            
        valid_starts = year_keys[:max_start_idx + 1]
        start = random.choice(valid_starts)
        start_idx = year_keys.index(start)
        window = year_keys[start_idx: start_idx + span]
        end = window[-1]

        if any(y not in years for y in window):
            fail_counter += 1
            continue

        all_entities = sum([years[y] for y in window], [])
        if not all_entities or not isinstance(all_entities[0], str):
            fail_counter += 1
            continue

        references = list(set(all_entities))
        if not (min_answers <= len(references) <= max_answers):
            fail_counter += 1
            continue
            
        if any(ref.startswith("Q") and ref[1:].isdigit() for ref in references):
            fail_counter += 1
            continue

        if start == end:
            question = question_template["single"].format(label=label, start=start)
        else:
            question = question_template["double"].format(label=label, start=start, end=end)

        if question in unique_keys:
            fail_counter += 1
            continue

        unique_keys.add(question)
        candidates.append({
            "question": question,
            "reference": references,
            "answer_count": len(references),
            "meta": {
                "entity_label": label,
                "entity_type": entity,
                "year_range": [int(start), int(end)]
            }
        })
        label_data_counter[label] = label_data_counter.get(label, 0) + 1
        pbar.update(1)
        fail_counter = 0

    pbar.close()
    return candidates

# ==================== Country Strategy ====================

def generate_country(facts, min_answers, max_answers, qa_count, question_template):
    candidates = []
    fail_counter = 0
    pbar = tqdm(total=qa_count, desc="Generating country")

    label_data_counter = {k: 0 for k in facts.keys()}
    unique_keys = set()
    
    FILTER_KEYWORDS = ["Province", "State", "Region", "County", "City", "Department", "District", "Division"]

    def clean_name(name):
        for kw in FILTER_KEYWORDS:
            name = name.replace(kw, "")
        return name.strip()

    def adjust_area(val, is_min=True):
        digits = len(str(abs(int(val))))
        delta = 1 if digits < 3 else 10 ** (digits - 2)
        return int(val) - delta if is_min else int(val) + delta

    while len(candidates) < qa_count:
        if fail_counter >= MAX_FAILS:
            print("❌ Too many failures. Exiting.")
            break
            
        country_name, divisions = random.choice(list(facts.items()))
        label = country_name
        
        if label_data_counter.get(label, 0) >= qa_count * ENTITY_FILTER_CONFIG["country"]:
            fail_counter += 1
            continue
            
        if not divisions or len(divisions) < min_answers:
            fail_counter += 1
            continue

        division_objs = []
        area_numbers = []
        for name, area_range in divisions.items():
            if not isinstance(area_range, list) or len(area_range) < 2:
                continue
            area_min, area_max = area_range[0], area_range[1]
            if area_min is None or area_max is None:
                continue
            try:
                area_min_int = adjust_area(area_min, is_min=True)
                area_max_int = adjust_area(area_max, is_min=False)
            except Exception:
                continue
            division_objs.append({
                "name": clean_name(str(name)),
                "area_min": area_min_int,
                "area_max": area_max_int
            })
            area_numbers.append(area_min_int)
            area_numbers.append(area_max_int)

        if len(division_objs) < min_answers:
            fail_counter += 1
            continue

        area_numbers = sorted(set(area_numbers))
        if len(area_numbers) < 2:
            fail_counter += 1
            continue

        area_min, area_max = sorted(random.sample(area_numbers, 2))
        if area_min == area_max:
            fail_counter += 1
            continue

        filtered_divisions = []
        has_partial_overlap = False
        
        for d in division_objs:
            # Strict inclusion check
            if d["area_min"] >= area_min and d["area_max"] <= area_max:
                filtered_divisions.append(d)
            # Fuzzy overlap check
            elif (d["area_min"] < area_min and d["area_max"] > area_min) or \
                 (d["area_min"] < area_max and d["area_max"] > area_max) or \
                 (d["area_min"] < area_min and d["area_max"] > area_max) or \
                 (d["area_min"] < area_max and d["area_max"] > area_min):
                has_partial_overlap = True
                break

        if has_partial_overlap:
            fail_counter += 1
            continue

        references = [d["name"] for d in filtered_divisions]
        references = list(dict.fromkeys(references)) # Deduplicate
        
        if any(ref.startswith("Q") and ref[1:].isdigit() for ref in references):
            fail_counter += 1
            continue

        if not (min_answers <= len(references) <= max_answers):
            fail_counter += 1
            continue

        unique_key = (country_name, area_min, area_max)
        if unique_key in unique_keys:
            fail_counter += 1
            continue
        unique_keys.add(unique_key)

        question = question_template["single"].format(
            country_name=country_name, area_min=area_min, area_max=area_max
        )
        
        candidates.append({
            "question": question,
            "reference": references,
            "answer_count": len(references),
            "meta": {
                "country": country_name,
                "area_min": area_min,
                "area_max": area_max,
                "entity_type": "country"
            }
        })
        label_data_counter[label] = label_data_counter.get(label, 0) + 1
        pbar.update(1)
        fail_counter = 0

    pbar.close()
    return candidates

# ==================== Language Strategy ====================

def generate_language(facts, min_answers, max_answers, qa_count, question_template):
    entity_pool = []
    for label, countries in facts.items():
        entity_pool.append((label, None, True))

    parents = [x for x in entity_pool if x[2]]
    children = [x for x in entity_pool if not x[2]]
    
    candidates = []
    fail_counter = 0
    unique_keys = set()
    pbar = tqdm(total=qa_count, desc="Generating language")

    label_data_counter = {k[0]: 0 for k in entity_pool}

    def get_country_names(label):
        return set(facts.get(label, []))

    while len(candidates) < qa_count:
        if fail_counter >= MAX_FAILS:
            print("❌ Too many failures. Exiting.")
            break

        pool = parents + children
        if len(pool) < 3:
            print("❌ Not enough languages.")
            break

        # Randomly sample 3, to artificially create duplicates (single/double), select one to be duplicated
        lang_samples = random.sample(pool, 3)
        lang1, lang2, lang3 = lang_samples
        # Force overlap for double/single template (per original logic)
        lang2 = lang1
        lang3 = lang1

        label1, qid1, is_parent1 = lang1
        label2, qid2, is_parent2 = lang2
        label3, qid3, is_parent3 = lang3

        if (label1.startswith("Q") and label1[1:].isdigit()) or \
           (label2.startswith("Q") and label2[1:].isdigit()) or \
           (label3.startswith("Q") and label3[1:].isdigit()):
            fail_counter += 1
            continue

        countries1 = get_country_names(label1)
        countries2 = get_country_names(label2)
        countries3 = get_country_names(label3)
        all_countries = list(dict.fromkeys([c for c in countries1.union(countries2).union(countries3) if c]))

        entity = "language"
        max_label_data = qa_count * ENTITY_FILTER_CONFIG[entity]
        if label_data_counter.get(label1, 0) >= max_label_data:
             fail_counter += 1
             continue

        if not (min_answers <= len(all_countries) <= max_answers):
            fail_counter += 1
            continue

        unique_key = frozenset([label1, label2, label3])
        if unique_key in unique_keys:
            fail_counter += 1
            continue
        unique_keys.add(unique_key)

        if label1 == label2 == label3:
            question = question_template["single"].format(label1=label1)
        elif label1 == label2 or label1 == label3 or label2 == label3:
            labels = list({label1, label2, label3})
            question = question_template["double"].format(label1=labels[0], label2=labels[1])
        else:
            question = question_template["triple"].format(label1=label1, label2=label2, label3=label3)

        candidates.append({
            "question": question,
            "reference": all_countries,
            "answer_count": len(all_countries),
            "meta": {
                "language1": label1,
                "language2": label2,
                "language3": label3,
                "entity_type": "language",
            }
        })

        label_data_counter[label1] = label_data_counter.get(label1, 0) + 1
        pbar.update(1)
        fail_counter = 0

    pbar.close()
    return candidates

# ==================== Office Strategy ====================

def generate_office(facts, min_answers, max_answers, qa_count, question_template):
    entity = "office"
    candidates = []
    fail_counter = 0
    unique_keys = set()
    pbar = tqdm(total=qa_count, desc=f"Generating {entity}")

    label_data_counter = {}
    valid_labels = [label for label in facts if not (label.startswith("Q") and label[1:].isdigit())]
    max_label_data = float(qa_count * ENTITY_FILTER_CONFIG.get(entity, 1.0))

    while len(candidates) < qa_count:
        if fail_counter >= MAX_FAILS:
            break
        if not valid_labels:
            break

        label = random.choice(valid_labels)
        if label_data_counter.get(label, 0) >= max_label_data:
            fail_counter += 1
            continue

        year_person_map = facts[label]
        year_keys = [y for y in year_person_map if y.isdigit() and int(y) >= 1800 and y not in ("2024", "2025")]
        year_keys = sorted(year_keys, key=int)
        
        if not year_keys:
            fail_counter += 1
            continue

        min_span = 1
        max_span = len(year_keys)
        span = random.randint(min_span, max_span)
        max_start_idx = len(year_keys) - span
        
        if max_start_idx < 0:
            fail_counter += 1
            continue
            
        valid_starts = year_keys[:max_start_idx + 1]
        start = random.choice(valid_starts)
        start_idx = year_keys.index(start)
        window = year_keys[start_idx : start_idx + span]
        end = window[-1]

        if any(y not in year_person_map for y in window):
            fail_counter += 1
            continue

        references = [year_person_map[y] for y in window]
        # Filter QID and flatten list (if references contains list nested)
        flat_refs = []
        for r in references:
            if isinstance(r, list):
                flat_refs.extend(r)
            else:
                flat_refs.append(r)
        
        references = [r for r in flat_refs if isinstance(r, str) and not (r.startswith("Q") and r[1:].isdigit())]
        references = list(set(references))

        if not (min_answers <= len(references) <= max_answers):
            fail_counter += 1
            continue

        if start == end:
            question = question_template["single"].format(label=label, start=start)
        else:
            question = question_template["double"].format(label=label, start=start, end=end)

        if question in unique_keys:
            fail_counter += 1
            continue

        unique_keys.add(question)
        candidates.append({
            "question": question,
            "reference": references,
            "answer_count": len(references),
            "meta": {
                "entity_label": label,
                "year_range": [int(start), int(end)],
                "entity_type": entity
            },
        })
        label_data_counter[label] = label_data_counter.get(label, 0) + 1
        pbar.update(1)
        fail_counter = 0

    pbar.close()
    return candidates

# ==================== River Strategy ====================

def generate_river(facts, min_answers, max_answers, qa_count, question_template):
    entity_pool = []
    for label in facts.keys():
        entity_pool.append((label, None, True))

    candidates = []
    fail_counter = 0
    unique_keys = set()
    pbar = tqdm(total=qa_count, desc="Generating river")

    label_data_counter = {k[0]: 0 for k in entity_pool}

    def get_country_names(label):
        return set(facts.get(label, []))

    while len(candidates) < qa_count:
        if fail_counter >= MAX_FAILS:
            break

        if len(entity_pool) < 3:
            print("❌ Not enough rivers.")
            break

        # Random selection logic: For generating single/double, artificially create duplicates
        river_samples = random.sample(entity_pool, 3)
        river1, river2, river3 = river_samples
        river2 = river1
        river3 = river1

        label1, qid1, is_parent1 = river1
        label2, qid2, is_parent2 = river2
        label3, qid3, is_parent3 = river3

        if (label1.startswith("Q") and label1[1:].isdigit()) or \
           (label2.startswith("Q") and label2[1:].isdigit()) or \
           (label3.startswith("Q") and label3[1:].isdigit()):
            fail_counter += 1
            continue

        countries1 = get_country_names(label1)
        countries2 = get_country_names(label2)
        countries3 = get_country_names(label3)
        all_countries = list(dict.fromkeys([c for c in countries1.union(countries2).union(countries3) if c]))

        entity = "river"
        max_label_data = qa_count * ENTITY_FILTER_CONFIG[entity]
        if label_data_counter.get(label1, 0) >= max_label_data:
             fail_counter += 1
             continue

        if not (min_answers <= len(all_countries) <= max_answers):
            fail_counter += 1
            continue

        unique_key = frozenset([label1, label2, label3])
        if unique_key in unique_keys:
            fail_counter += 1
            continue
        unique_keys.add(unique_key)

        if label1 == label2 == label3:
            question = question_template["single"].format(label1=label1)
        elif label1 == label2 or label1 == label3 or label2 == label3:
            labels = list({label1, label2, label3})
            question = question_template["double"].format(label1=labels[0], label2=labels[1])
        else:
            question = question_template["triple"].format(label1=label1, label2=label2, label3=label3)

        candidates.append({
            "question": question,
            "reference": all_countries,
            "answer_count": len(all_countries),
            "meta": {
                "river1": label1,
                "river2": label2,
                "river3": label3,
                "entity_type": "river"
            }
        })

        label_data_counter[label1] = label_data_counter.get(label1, 0) + 1
        pbar.update(1)
        fail_counter = 0

    pbar.close()
    return candidates

# ==================== Main Flow ====================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate QA data for multiple entities')
    parser.add_argument('--min_answers', type=int, default=6, help='Minimum number of answers')
    parser.add_argument('--max_answers', type=int, default=6, help='Maximum number of answers')
    parser.add_argument('--qa_count', type=int, default=500, help='Number of QA pairs per entity')
    parser.add_argument('--entity', type=str, default=None, help='Specific entity type to process (default: all)')
    parser.add_argument('--input_dir', type=str, default=None, help='Input directory containing fact JSONs')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for JSONL files')
    return parser.parse_args()

def generate_for_entity(entity, min_answers, max_answers, qa_count, input_dir, output_dir):
    config = ENTITY_CONFIG[entity]
    question_template = config["question_template"]
    
    if entity == "math":
        return generate_math(min_answers, max_answers, qa_count, output_dir)
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory {input_dir} does not exist. Skipping {entity}.")
        return

    FACTS_FILE = f"{input_dir}/{entity}_structured_facts.json"
    if not os.path.exists(FACTS_FILE):
        print(f"❌ Fact file {FACTS_FILE} not found. Skipping {entity}.")
        return
       
    entity_output_dir = os.path.join(output_dir, entity)
    os.makedirs(entity_output_dir, exist_ok=True)
    OUTPUT_FILE = f"{entity_output_dir}/qa_{entity}_{min_answers}_{max_answers}_{qa_count}.jsonl"

    facts = json.load(open(FACTS_FILE, encoding="utf-8"))
    print(f"\n=== Generating for entity: {entity} ===")
    print(f"Loaded {len(facts)} facts")
    
    generate_map = {
        "award": generate_award,
        "region": generate_country,
        "language": generate_language,
        "office": generate_office,
        "river": generate_river
    }
    
    generator = generate_map.get(entity)
    if not generator:
        print(f"❌ No generator found for {entity}")
        return

    candidates = generator(facts, min_answers, max_answers, qa_count, question_template)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for item in candidates:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Finished {entity}. {len(candidates)} QA written to {OUTPUT_FILE}")

def main():
    args = parse_arguments()
    min_answers = args.min_answers
    max_answers = args.max_answers
    qa_count = args.qa_count
    target_entity = args.entity
    
    print("Current Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
        
    if target_entity != 'all':
        if target_entity not in ENTITY_CONFIG:
            print(f"❌ Unsupported entity type: {target_entity}")
            print(f"Supported: {list(ENTITY_CONFIG.keys())}")
            return
        generate_for_entity(target_entity, min_answers, max_answers, qa_count, args.input_dir, args.output_dir)
    else:
        for entity in ENTITY_CONFIG.keys():
            generate_for_entity(entity, min_answers, max_answers, qa_count, args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()